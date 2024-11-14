from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.models.grpo.head import GRPOHead
from fast_llm.models.grpo.config import GRPOBaseModelConfig, GRPOModelConfig
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel


class GRPOBaseModel(GPTBaseModel):
    _config: GRPOBaseModelConfig
    config_cls = GRPOBaseModelConfig

    def __init__(
        self,
        config: GRPOModelConfig,
        distributed_config: DistributedConfig,
    ):
        super().__init__(config, distributed_config)
        assert self._config.transformer.use_rotary_position_embeddings
        assert not self._config.use_absolute_position_embeddings

    def get_layers(self):
        return [
            LanguageModelEmbedding(self._config, self._tensor_space),
            *[
                TransformerLayer(
                    self._config.transformer,
                    self._tensor_space,
                    layer_index=i + 1,
                )
                for i in range(self._config.transformer.num_layers)
            ],
            GRPOHead(self._config, self._tensor_space),  # Use our custom head
        ]

    def preprocess(
        self,
        batch: dict,
        preprocessed_meta=None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics=None
    ):
        # Extract GRPO specific inputs
        grpo_inputs = {
            "rewards": batch.pop("rewards")[:, 1:],
            "advantages": batch.pop("advantages")[:, 1:],
            "ref_logprobs": batch.pop("ref_logprobs")[:, 1:],
            "old_logprobs": batch.pop("old_logprobs")[:, 1:],
            "grpo_config": self._config.grpo,
        }
        
        # Process the remaining inputs using parent class
        preprocessed = super().preprocess(
            batch["input_ids"], 
            preprocessed_meta,
            phase=phase,
            iteration=iteration,
            metrics=metrics
        )
        
        # Add GRPO inputs to kwargs
        for tokens, kwargs in preprocessed:
            kwargs.update(grpo_inputs)
            
        return preprocessed

    @property
    def loss_defs(self):
        # TODO: Adjust or reimplement.
        return super().loss_defs


class GRPOModel(GPTModel):
    config_class = GRPOModelConfig
    base_model_class = GRPOBaseModel
