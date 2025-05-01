from lm_eval.__main__ import cli_evaluate

from fast_llm.models.ssm.external.eval.apriel_eval_wrapper import AprielSSMWrapper  # noqa: F401

if __name__ == "__main__":
    cli_evaluate()
