import logging
import sys
import traceback

from fast_llm.config import ValidationError
from fast_llm.engine.config_utils.logging import configure_logging
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.config_utils.runnable import RunnableConfig

# Import these submodules to ensure classes are added to the dynamic class registry.
import fast_llm.data.auto  # isort: skip
import fast_llm.engine.checkpoint.convert  # isort: skip
import fast_llm.models.auto  # isort: skip

logger = logging.getLogger(__name__)


def fast_llm_main(args=None):
    # TODO: Add hook to register model classes? (environment variable?)
    # (Pre-)configure logging
    configure_logging()
    try:
        if args is None:
            args = sys.argv[1:]
        # TODO: Remove backward compatibility.
        if len(args) >= 2 and args[0] == "train":
            if args[1] == "gpt":
                args = ["type=train_gpt"] + args[2:]
            elif args[1] == "hybrid_ssm":
                args = ["type=train_hybrid_ssm"] + args[2:]
        elif len(args) >= 2 and args[0] == "convert":
            if "=" not in args[1]:
                args = ["type=convert", f"model={args[1]}"] + args[2:]
        elif len(args) >= 2 and args[0] == "prepare" and args[1] == "gpt_memmap":
            args = ["type=prepare_gpt_memmap"] + args[2:]
        RunnableConfig.parse_and_run(args)
    except Exception as e:
        if sys.gettrace():
            raise
        if isinstance(e, ValidationError):
            log_main_rank(traceback.format_exc(), log_fn=logger.error)
        else:
            logger.critical(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    fast_llm_main()
