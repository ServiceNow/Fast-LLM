import logging
import logging.config
import math
import pathlib

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.utils import Assert

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


@config_class()
class TensorLogsConfig(Config):
    save: bool = Field(
        default=False,
        desc="Save tensor logs to an artifact file.",
        hint=FieldHint.logging,
    )
    show: bool = Field(
        default=True,
        desc="Post all tensor logs to stdout. May lead to extremely large log",
        hint=FieldHint.logging,
    )
    max_elements: int = Field(
        default=8,
        desc="Maximum number of tensor values to print for each tensor when posting tensor logs to stdout.",
        hint=FieldHint.logging,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )


class TensorLogs:
    # A global buffer for holding logged tensor stats.
    _tensor_log_stats: list | None = None
    config: TensorLogsConfig | None = None

    @classmethod
    def reset(cls, config: TensorLogsConfig):
        cls.config = config
        cls._tensor_log_stats = [] if config.save else None

    @classmethod
    def append(cls, stats):
        if cls._tensor_log_stats is not None:
            cls._tensor_log_stats.append(stats)

    @classmethod
    def get(cls):
        return cls._tensor_log_stats
