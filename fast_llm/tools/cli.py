import argparse
import logging
import sys
import traceback

from fast_llm.config import ValidationError
from fast_llm.engine.config_utils.logging import configure_logging
from fast_llm.engine.config_utils.run import log_main_rank

logger = logging.getLogger(__name__)


def fast_llm(args=None):
    # TODO: Add hook to register model classes? (environment variable?)
    # (Pre-)configure logging
    configure_logging()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("subcommand", choices=["train", "convert"], help="The Fast-LLM command to run")
    parsed, unparsed = parser.parse_known_args(args)
    try:
        if parsed.subcommand == "train":
            from fast_llm.tools.train import train

            train(unparsed)
        elif parsed.subcommand == "convert":
            from fast_llm.tools.convert import convert

            convert(unparsed)
        else:
            raise RuntimeError("Unknown subcommand")
    except ValidationError:
        log_main_rank(traceback.format_exc(), log_fn=logger.error)
        sys.exit(1)
    except Exception:  # noqa
        logger.critical(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    fast_llm()
