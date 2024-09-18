import argparse

from fast_llm.config import NoAutoValidate
from fast_llm.distributed import Distributed
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.engine.training.trainer import Trainer
from fast_llm.models.auto import trainer_registry


def train_base(trainer_cls: type["Trainer"], *, args=None):
    with NoAutoValidate():
        config: TrainerConfig = trainer_cls.config_class.from_args(args)
    try:
        config.validate()
        distributed = Distributed(config.distributed)
        run = config.get_run(distributed)
        trainer = trainer_cls(config=config)
    except Exception:
        # Logging may not have been configured yet,
        # but we still want to show the config for debugging.
        config.show_main_rank(config.model.distributed, log_fn=print)
        raise
    finally:
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
    parsed, unparsed = parser.parse_known_args(args)
    trainer_cls = trainer_registry[parsed.model_type]
    train_base(trainer_cls, args=unparsed)


if __name__ == "__main__":
    train()
