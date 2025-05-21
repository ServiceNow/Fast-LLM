# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Apriel SSM model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import is_torch_available, logging

logger = logging.get_logger(__name__)

if is_torch_available():
    pass


class AprielSSMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AprielModel`]. It is used to instantiate an Apriel
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Apriel-5B-Base.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        ....
    ```"""

    model_type = "apriel_ssm"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        hidden_act="silu",
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        mlp_bias=False,
        rms_norm_eps=1e-5,
        ssm_cfg: dict = None,
        head_dim: int = 128,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        # self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        # self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        # self.rope_theta = rope_theta
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        # if self.rope_scaling is not None and "type" in self.rope_scaling:
        #     self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.ssm_cfg = ssm_cfg or {
            "d_state": 64,
            "n_v_heads": 24,
            "n_qk_heads": 24,
            "expand": 1,
            "chunk_size": 128,
            "activation": "identity",
            "bias": False,
            "d_inner": 24 * self.head_dim,  # num_heads * head_dim
        }
        if self.head_dim != self.ssm_cfg["d_inner"] // self.ssm_cfg["n_qk_heads"]:
            logger.warning("Head dim is not equal to d_inner // n_qk_heads.")


__all__ = ["AprielConfig"]
