import logging
import logging.config
import math
import pathlib
import typing

logger = logging.getLogger(__name__)


def configure_logging(
    *,
    log_timestamps: bool = True,
    enable_all_loggers: bool = False,
    rank: int = 0,
    world_size: int = 1,
    directory: pathlib.Path | str | None = None,
):
    rank_str = str(rank).zfill(math.ceil(math.log10(world_size)))
    format_ = f"{f'%(asctime)s ' if log_timestamps else ''}{'' if world_size==1 else f'[Rank {rank_str}] '}%(message)s"
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": format_,
                "use_colors": True,
            }
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "fast_llm": {"level": "INFO"},
            "__main__": {"level": "INFO"},
        },
        "root": {"handlers": ["default"], "level": "INFO" if enable_all_loggers else "WARNING"},
    }
    if directory is not None:
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        logging_config["handlers"]["file"] = {
            "level": "INFO",
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": directory / f"logs_rank_{rank_str}.txt",
        }
        logging_config["root"]["handlers"].append("file")
    logging.config.dictConfig(logging_config)


def log(*message, log_fn: typing.Union[BaseException, typing.Callable] = logger.info, join: str = ", "):
    message = join.join([str(m() if callable(m) else m) for m in message])
    if isinstance(log_fn, BaseException):
        raise log_fn(message)
    else:
        return log_fn(message)


class TensorLogs:
    # A global buffer for holding logged tensor stats.
    _tensor_log_stats: list | None = None
    max_logged_elements = 8
    verbose: bool = True

    @classmethod
    def reset(cls, enabled=True):
        cls._tensor_log_stats = [] if enabled else None

    @classmethod
    def enabled(cls):
        return cls._tensor_log_stats is not None

    @classmethod
    def append(cls, stats):
        if cls._tensor_log_stats is not None:
            cls._tensor_log_stats.append(stats)

    @classmethod
    def get(cls):
        return cls._tensor_log_stats
