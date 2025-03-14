import logging
import typing

from fast_llm.data.data.gpt.data import GPTData
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.training.trainer import Trainer
from fast_llm.models.gpt.config import GPTTrainerConfig
from fast_llm.models.gpt.model import GPTModel

logger = logging.getLogger(__name__)


class GPTTrainer[ConfigType: GPTTrainerConfig](Trainer[ConfigType]):
    config_class: typing.ClassVar[type[GPTTrainerConfig]] = GPTTrainerConfig
    model_class: typing.ClassVar[type[GPTModel]] = GPTModel

    def _get_data(self) -> GPTData:
        return GPTData(
            config=self._config.data,
            distributed_config=self._config.model.distributed,
            vocab_size=self._config.model.base_model.vocab_size,
            max_sequence_length=self._config.batch.sequence_length,
            cross_document_attention=self._config.model.base_model.transformer.cross_document_attention,
        )

    def get_tflops(self, phase: PhaseType, elapsed_time_per_iteration) -> tuple[int, int]:
        # TODO: Do in model, automate/generalize, get other stats
        """Get tflop/s/GPU from global-batch-size and elapsed-time"""
        checkpoint_activations_factor = 3 if phase == PhaseType.training else 1
        transformer_config = self._config.model.base_model.transformer
        sequence_length = self._config.batch.sequence_length

        tokens = self._config.batch.batch_size * sequence_length
        transformer_flops_base = 2 * checkpoint_activations_factor * tokens * transformer_config.num_layers
        dense_flops_base = transformer_flops_base * transformer_config.hidden_size
        # Query, key, value, dense.
        flops_per_iteration = (
            2
            * (transformer_config.num_attention_heads + transformer_config.head_groups)
            * transformer_config.kv_channels
            * dense_flops_base
        )
        # MLP
        flops_per_iteration += (
            (2 + transformer_config.gated)
            * transformer_config.ffn_hidden_size
            * dense_flops_base
            * transformer_config.num_experts_per_token
        )

        # LM-head
        flops_per_iteration += 6 * tokens * transformer_config.hidden_size * self._config.model.base_model.vocab_size

        # Attention-matrix computation
        attn_flops_base = transformer_flops_base * transformer_config.projection_size
        if transformer_config.window_size is None:
            # Ignore masked values (s**2/2)
            attn_flops = attn_flops_base * sequence_length
            model_tflops = flops_per_iteration + attn_flops
        else:
            # s*w - w**2/2
            attn_flops = (
                2
                * attn_flops_base
                * transformer_config.window_size
                * (1 - transformer_config.window_size / 2 / sequence_length)
            )
            model_tflops = flops_per_iteration + attn_flops

        # Partial recomputation (normal is 2 ops * ckpt_factor = 6, adding 1 for recomputing Q x K)
        hardware_flops = flops_per_iteration + 7 / 6 * attn_flops
        ratio = elapsed_time_per_iteration * self._config.model.distributed.world_size * 1e12
        return model_tflops / ratio, hardware_flops / ratio
