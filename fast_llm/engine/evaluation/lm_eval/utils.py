import argparse
import json
import logging
import os
import pathlib
import sys
from pathlib import Path

import lm_eval.__main__
import lm_eval.evaluator
import lm_eval.loggers
import lm_eval.tasks
import lm_eval.utils

from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


def parse_eval_args(parser: argparse.ArgumentParser, args: list[str]) -> argparse.Namespace:
    lm_eval.__main__.check_argument_types(parser)
    return parser.parse_args(args)


def prepare_lm_eval_simple_eval_params(
    cli_args: list[str],
    completed_steps: int,
    run_index: int,
) -> tuple[argparse.Namespace, dict[str, any]]:
    """
    Parses CLI arguments for an LM evaluation run and prepares keyword arguments
    for the `evaluate` function.

    This function wraps argument parsing, environment configuration, task resolution,
    and metadata setup needed for evaluation with Fast-LLM's `lm_eval` wrapper. It also
    handles special cases like hub token injection, dynamic sample loading, and task
    listing commands.

    Args:
        cli_args (list[str]): Command-line arguments, excluding the program name.
        completed_steps (int): Current number of completed training steps, used to
            uniquely tag evaluation output paths.
        run_index (int): index of the current run of Fast-LLM experiment

    Returns:
        tuple:
            - argparse.Namespace: Parsed CLI arguments.
            - dict: Keyword arguments to pass into `simple_evaluate`, including task list,
              tracker, cache settings, random seeds, and generation parameters.

    Raises:
        ValueError: If required fields like `--tasks` or `--output_path` are missing
                    when needed, or if misconfigured combinations are detected.
        SystemExit: If special task listing flags are used.
    """
    parser = lm_eval.__main__.setup_parser()
    parser.add_argument(
        "--no_defaults",
        action="store_true",
    )
    args = parse_eval_args(parser, cli_args)

    # NOTE: all this args are set by fast_llm on the model directly or not used here
    Assert.eq(args.wandb_args, "")
    Assert.eq(args.wandb_config_args, "")
    Assert.eq(args.model, "hf")
    Assert.eq(args.model_args, "")
    Assert.eq(int(args.batch_size), 1)
    Assert.none(args.max_batch_size)
    Assert.none(args.device)

    # update the evaluation tracker args with the output path and the HF token
    evaluation_tracker_args = ""
    if args.output_path:
        args.output_path = str(pathlib.Path(args.output_path) / f"runs/{run_index}/{completed_steps}")
        evaluation_tracker_args += f",output_path={args.output_path}"

    evaluation_tracker_args = lm_eval.utils.simple_parse_args_string(evaluation_tracker_args)
    evaluation_tracker = lm_eval.loggers.EvaluationTracker(**evaluation_tracker_args)

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError(
            "When `fewshot_as_multiturn` is selected, `apply_chat_template` must be set (either to `True` or to the chosen template name)."
        )

    if args.include_path is not None:
        args.include_path = args.include_path.split(",")
        logger.info(f"Including paths: {args.include_path}")
    metadata = (
        lm_eval.utils.simple_parse_args_string(args.model_args)
        if isinstance(args.model_args, str)
        else args.model_args if isinstance(args.model_args, dict) else {}
    ) | (args.metadata if isinstance(args.metadata, dict) else lm_eval.utils.simple_parse_args_string(args.metadata))

    task_manager = lm_eval.tasks.TaskManager(
        verbosity=args.verbosity,
        include_path=args.include_path,
        include_defaults=not args.no_defaults,
        metadata=metadata,
    )

    if args.limit:
        logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")
    if args.samples:
        assert args.limit is None, "If --samples is not None, then --limit must be None."
        if (samples := Path(args.samples)).is_file():
            args.samples = json.loads(samples.read_text())
        else:
            args.samples = json.loads(args.samples)

    if args.tasks is None:
        logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        print(task_manager.list_all_tasks())
        sys.exit()
    elif args.tasks == "list_groups":
        print(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        print(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        print(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = lm_eval.utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = lm_eval.utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{lm_eval.utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all"
                    " available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG'"
                    " to troubleshoot task registration issues."
                )

    logger.info(f"Selected Tasks: {task_names}")

    request_caching_args = lm_eval.evaluator.request_caching_arg_to_dict(cache_requests=args.cache_requests)

    eval_kwargs = dict(
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        # batch_size=args.batch_size,
        # max_batch_size=args.max_batch_size,
        # device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        samples=args.samples,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=args.system_instruction,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
        metadata=metadata,
        **request_caching_args,
    )

    return args, eval_kwargs


def process_lm_eval_results(
    args: argparse.Namespace,
    results: dict[str, any],
    evaluation_tracker: lm_eval.loggers.EvaluationTracker,
    completed_steps: int | None,
) -> None:
    if results is not None:
        completed_steps = 0 if completed_steps is None else completed_steps
        import wandb

        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=lm_eval.utils.handle_non_serializable, ensure_ascii=False)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        # Add W&B logging if we have the run to log to
        #  we expect the rest of the fast_llm code will finish the run.
        if wandb.run is not None:
            try:
                wandb_logger = lm_eval.loggers.WandbLogger(init_args={"step": completed_steps})
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if args.log_samples:
                    wandb_logger.log_eval_samples(samples)
            except Exception as e:
                logger.info(f"Logging to Weights and Biases failed due to {e}")

        evaluation_tracker.save_results_aggregated(results=results, samples=samples if args.log_samples else None)

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

        if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
            evaluation_tracker.recreate_metadata_card()

        # TODO: convert to logging entries instead?
        print(
            f"{results["config"]["model"]}, gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {results["config"]["batch_size"]}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(lm_eval.utils.make_table(results))
        if "groups" in results:
            print(lm_eval.utils.make_table(results, "groups"))
