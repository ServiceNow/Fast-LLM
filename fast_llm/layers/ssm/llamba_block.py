import typing

from fast_llm.layers.transformer.transformer import BaseBlock

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_space import TensorSpace
    from fast_llm.layers.ssm.config import SSMConfig
    from fast_llm.layers.transformer.config import TransformerConfig


class LlambaBlock(BaseBlock):
    """
    A transformer-like decoder block with a SSM mixer, see https://arxiv.org/abs/2502.14458
    """

    name = "Llamba block"
    _mixer_module_name = "mixer"

    def __init__(
        self,
        config_transformer: "TransformerConfig",
        config_ssm: "SSMConfig",
        tensor_space: "TensorSpace",
        mixer_cls,
        layer_index: int,
        return_input: bool = False,
    ):
        self.mixer_cls = mixer_cls
        self._config_ssm = config_ssm
        self._debug_mode = self._config_ssm.debug_ssm
        super().__init__(config_transformer, tensor_space, layer_index, return_input)

    def _create_mixer(self):
        self.mixer = self.mixer_cls(self._config_ssm, layer_idx=self._layer_index, tensor_space=self._tensor_space)

    # def forward(
    #     self,
    #     input_: torch.Tensor,
    #     kwargs: dict[str, typing.Any],
    #     losses: dict[str, typing.Any] | None = None,
    #     metrics: dict[str, typing.Any] | None = None,
    # ) -> torch.Tensor:
    #     if isinstance(input_, TensorMeta):
    #         return self._get_meta(input_, "output", kwargs)
    #     generator = (
    #         self._tensor_space.distributed.tp_generator
    #         if self._tensor_space.distributed_config.sequence_tensor_parallel
    #         else self._tensor_space.distributed.pp_generator
    #     )
    #     residual = input_
    #     if self._debug_mode:
    #         self._debug_log(None, "Begin", kwargs)
    #     fw_input = input_
    #     hidden_states = self.norm_1(input_)
    #     if self._debug_mode:
    #         self._debug_log(hidden_states, "Norm 1", kwargs)
    #     hidden_states, _ = getattr(self, self._mixer_module_name)(hidden_states, kwargs)
    #     hidden_states = self.norm_2(input_)
    #     if self._debug_mode:
    #         self._debug_log(hidden_states, "Norm 2", kwargs)
    #     hidden_states, bias = self.mlp(hidden_states, kwargs, losses, metrics)
    #     if self._debug_mode:
    #         self._debug_log(hidden_states, "MLP output", kwargs, bias=bias)
    #     with set_generator(generator):
    #         hidden_states = self._bias_dropout_add(hidden_states, bias, input_)
    #     if self._debug_mode:
    #         self._debug_log(None, "MLP residual", kwargs, bias=bias)
    #     if self._return_input:
    #         hidden_states = torch.stack((fw_input, hidden_states), dim=0)
    #     return hidden_states
