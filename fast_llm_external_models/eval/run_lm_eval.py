from lm_eval.__main__ import cli_evaluate

from fast_llm_external_models.eval.apriel_eval_wrapper import (  # noqa: F401
    AprielHybrid15bSSMWrapper,
    AprielHybridSSMWrapper,
    AprielSSMWrapper,
)

if __name__ == "__main__":
    cli_evaluate()
