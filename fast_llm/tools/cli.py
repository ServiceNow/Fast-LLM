import argparse
import logging
import sys
import traceback

logger = logging.getLogger(__name__)


def fast_llm(args=None):
    # TODO: Add hook to register model classes? (environment variable?)
    parser = argparse.ArgumentParser()
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
    except Exception:  # noqa
        logger.critical(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    fast_llm()
