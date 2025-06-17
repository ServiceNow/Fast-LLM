import typing

from fast_llm.config import config_class
from fast_llm.engine.config_utils.runnable import RunnableConfig


@config_class(registry=True, dynamic_type={RunnableConfig: "evaluate"})
class EvaluatorsConfig(RunnableConfig):
    _abstract = True

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        if "training" in default:
            default["training"]["train_iters"] = 0
        else:
            default["training"] = {"train_iters": 0}
        return super()._from_dict(default, strict, flat)
