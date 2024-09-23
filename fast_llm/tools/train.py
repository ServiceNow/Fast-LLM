import argparse

from fast_llm.config import NoAutoValidate
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.models.auto import trainer_registry


def train_base(trainer_config_class: type["TrainerConfig"], *, args=None, do_run: bool = True):
    with NoAutoValidate():
        config: TrainerConfig = trainer_config_class.from_args(args)
    try:
        config.validate()
        if not do_run:
            return
        from fast_llm.engine.distributed.distributed import Distributed

        distributed = Distributed(config.distributed)
        run = config.get_run(distributed)
        trainer = trainer_config_class.get_trainer_class()(config=config)
    finally:
        # We always want to show the config for debugging.
        config.show_main_rank(config.model.distributed)
    with run:
        trainer.setup(distributed, run)
        trainer.run()


def train(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_type",
        choices=trainer_registry.keys(),
        help="The Fast-LLM model type to use. Must be defined in the trainer registry in `fast_llm.models.auto`.",
    )
    parser.add_argument(
        "-v",
        "--validate",
        dest="do_run",
        action="store_false",
        help="Validate the config without running the actual experiment.",
    )
    parsed, unparsed = parser.parse_known_args(args)
    trainer_cls = trainer_registry[parsed.model_type]
    train_base(trainer_cls, args=unparsed, do_run=parsed.do_run)


if __name__ == "__main__":
    train()
