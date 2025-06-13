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


def fast_llm_main(args: list[str] | None = None):
    # TODO: Add hook to register model classes? (environment variable?)
    # (Pre-)configure logging
    configure_logging()
    try:
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
