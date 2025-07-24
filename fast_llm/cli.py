import contextlib
import logging
import sys
import traceback

from fast_llm.config import ValidationError
from fast_llm.engine.config_utils.logging import configure_logging
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.utils import set_global_variables

# Import these submodules to ensure classes are added to the dynamic class registry.
import fast_llm.data.auto  # isort: skip
import fast_llm.engine.checkpoint.convert  # isort: skip
import fast_llm.models.auto  # isort: skip

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def fast_llm_main_wrapper():
    # (Pre-)configure logging
    configure_logging()
    # Set global and environment variables before third-party imports.
    set_global_variables()
    try:
        yield
    except Exception as e:
        if sys.gettrace():
            raise
        if isinstance(e, ValidationError):
            log_main_rank(traceback.format_exc(), log_fn=logger.error)
        else:
            logger.critical(traceback.format_exc())
        sys.exit(1)


def fast_llm_main(args: list[str] | None = None):
    # TODO: Add hook to register model classes? (environment variable?)
    with fast_llm_main_wrapper():
        RunnableConfig.parse_and_run(args)


if __name__ == "__main__":
    fast_llm_main()
