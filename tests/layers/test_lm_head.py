import collections
import dataclasses
import typing

import pytest
import torch

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.language_model.config import LM_HEAD_LOSS_NAME, LanguageModelKwargs
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.layers.language_model.loss.config import LanguageModelLossKwargs
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.utils import Assert
from tests.layers.test_lm_losses import reference_grpo_loss, reference_grpo_metrics, reference_gspo_loss
from tests.utils.utils import get_base_model, get_stage

NUM_TOKENS = 200
HIDDEN_SIZE = 256
VOCAB_SIZE = 500
GSPO_NUM_DOCUMENTS = 4


@dataclasses.dataclass
class LMHeadTestConfig:
    name: str
    label_loss: bool | float = False
    distillation_loss: bool | float = False
    distillation_temperature: float = 1.0
    z_loss: bool | float = False
    grpo_loss: bool | float = False
    gspo_loss: bool | float = False
    logits_scale_factor: float = 1.0
    final_logit_softcap: float | None = None
    compute_dtype: DataType = DataType.float32
    full_precision_residual: bool = False
    loss_masking: bool = False
    prediction_heads: int = 1
    tied_embedding_weight: bool = False
    num_splits: int = 1
    gspo_document_lengths: tuple[int, ...] | None = None
    loss_implementation: str = "per_loss"
    grpo_metrics: str | None = None

    @property
    def actual_label_loss(self):
        return (
            True
            if self.label_loss is False
            and self.distillation_loss is False
            and self.z_loss is False
            and self.grpo_loss is False
            and self.gspo_loss is False
            else self.label_loss
        )

    def get_config(self) -> GPTModelConfig:
        head_config = {
            "normalization": {"type": "rms_norm"},
            "logits_scale_factor": self.logits_scale_factor,
            "cross_entropy_splits": self.num_splits,
            "prediction_heads": self.prediction_heads,
        }
        if self.loss_implementation != "per_loss":
            head_config["loss_implementation"] = self.loss_implementation
        if self.final_logit_softcap is not None:
            head_config["final_logit_softcap"] = self.final_logit_softcap
        losses = {}
        if self.label_loss is not False:
            losses["label"] = {"type": "label"}
            if isinstance(self.label_loss, float):
                losses["label"]["weight"] = self.label_loss
        if self.distillation_loss is not False:
            losses["distillation"] = {"type": "distillation", "reference_model": "distillation"}
            if self.distillation_temperature != 1.0:
                losses["distillation"]["temperature"] = self.distillation_temperature
            if isinstance(self.distillation_loss, float):
                losses["distillation"]["weight"] = self.distillation_loss
        if self.z_loss is not False:
            losses["z_loss"] = {"type": "z_loss"}
            if isinstance(self.z_loss, float):
                losses["z_loss"]["weight"] = self.z_loss
        if self.grpo_loss is not False:
            losses["grpo_loss"] = {"type": "grpo"}
            if isinstance(self.grpo_loss, float):
                losses["grpo_loss"]["weight"] = self.grpo_loss
            if self.grpo_metrics is not None:
                losses["grpo_loss"]["metrics"] = self.grpo_metrics
        if self.gspo_loss is not False:
            losses["gspo_loss"] = {"type": "gspo"}
            if isinstance(self.gspo_loss, float):
                losses["gspo_loss"]["weight"] = self.gspo_loss
        if losses:
            head_config["losses"] = losses

        return GPTModelConfig.from_dict(
            {
                "base_model": {
                    "decoder": {"num_blocks": 0},
                    "embeddings": {"vocab_size": VOCAB_SIZE, "full_precision_residual": self.full_precision_residual},
                    "head": head_config,
                    "hidden_size": HIDDEN_SIZE,
                    "tied_embedding_weight": self.tied_embedding_weight,
                },
                "distributed": {"compute_dtype": self.compute_dtype, "use_cuda": torch.cuda.is_available()},
            },
        )

    @property
    def actual_gspo_document_lengths(self) -> tuple[int, ...]:
        if self.gspo_document_lengths is not None:
            Assert.eq(sum(self.gspo_document_lengths), NUM_TOKENS)
            return self.gspo_document_lengths
        document_length = NUM_TOKENS // GSPO_NUM_DOCUMENTS
        Assert.eq(document_length * GSPO_NUM_DOCUMENTS, NUM_TOKENS)
        return (document_length,) * GSPO_NUM_DOCUMENTS

    def get_inputs(self) -> tuple[torch.Tensor, dict[str, typing.Any]]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ = torch.randn(
            (NUM_TOKENS, HIDDEN_SIZE),
            dtype=(torch.float32 if self.full_precision_residual else self.compute_dtype.torch),
            device=device,
            requires_grad=True,
        )
        kwargs: dict[str, typing.Any] = {
            AttentionKwargs.grad_output: 1.0,
        }
        if self.loss_masking:
            kwargs[LanguageModelKwargs.loss_mask] = [
                torch.randint(0, 2, (NUM_TOKENS,), dtype=torch.bool, device=device)
                for _ in range(self.prediction_heads)
            ]
            kwargs[LanguageModelKwargs.num_labels_in_batch] = [
                loss_mask.sum().item() for loss_mask in kwargs[LanguageModelKwargs.loss_mask]
            ]
        else:
            kwargs[LanguageModelKwargs.num_labels_in_batch] = [NUM_TOKENS for _ in range(self.prediction_heads)]
        if self.actual_label_loss is not False or self.grpo_loss is not False or self.gspo_loss is not False:
            labels = [
                torch.randint(
                    0,
                    VOCAB_SIZE,
                    (NUM_TOKENS,),
                    dtype=torch.int64,
                    device=device,
                )
                for _ in range(self.prediction_heads)
            ]
            if LanguageModelKwargs.loss_mask in kwargs:
                labels = [
                    torch.where(mask, labels_, -100)
                    for labels_, mask in zip(labels, kwargs[LanguageModelKwargs.loss_mask], strict=True)
                ]
            kwargs[LanguageModelKwargs.labels] = labels

        if self.distillation_loss is not False:
            assert self.prediction_heads == 1
            kwargs[f"reference_distillation_hidden_states"] = {
                "head.logits": torch.randn(
                    input_.shape[:-1] + (VOCAB_SIZE,),
                    dtype=input_.dtype,
                    device=device,
                )
            }
        if self.grpo_loss is not False or self.gspo_loss is not False:
            kwargs[LanguageModelLossKwargs.advantages] = [
                torch.randn(input_.shape[:-1], dtype=torch.float32, device=device)
                for _ in range(self.prediction_heads)
            ]
            kwargs[LanguageModelLossKwargs.old_log_probabilities] = [
                torch.randn(input_.shape[:-1], dtype=torch.float32, device=device)
                for _ in range(self.prediction_heads)
            ]
            kwargs[LanguageModelLossKwargs.label_counts] = [
                torch.full(input_.shape[:-1], float((labels_ >= 0).sum()), dtype=torch.float32, device=device)
                for labels_ in kwargs[LanguageModelKwargs.labels]
            ]
            kwargs[LanguageModelKwargs.num_documents_in_batch] = (
                len(self.actual_gspo_document_lengths) if self.gspo_loss is not False else 1
            )
        if self.gspo_loss is not False:
            document_lengths = self.actual_gspo_document_lengths
            document_length_repeats = torch.tensor(document_lengths, dtype=torch.int64, device=device)
            kwargs[BlockKwargs.global_document_index_q] = torch.repeat_interleave(
                torch.arange(1, len(document_lengths) + 1, dtype=torch.int32, device=device), document_length_repeats
            )
            kwargs[BlockKwargs.num_documents_in_sequence] = len(document_lengths)
            kwargs[BlockKwargs.lengths] = list(document_lengths)
            # Override label_counts: per-token broadcast of the containing document's masked-label count
            # (the kernel's per-document `new_logprobs` aggregation depends on this).
            kwargs[LanguageModelLossKwargs.label_counts] = [
                torch.cat(
                    [
                        torch.full(
                            (document_length,),
                            float(document_mask.sum()),
                            dtype=torch.float32,
                            device=device,
                        )
                        for document_mask, document_length in zip(
                            torch.split(labels_ >= 0, document_lengths), document_lengths, strict=True
                        )
                    ]
                )
                for labels_ in kwargs[LanguageModelKwargs.labels]
            ]
        return input_, kwargs

    def get_reference_outputs(
        self,
        head: LanguageModelHead,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        tied_logit_weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        # Get reference outputs and grads
        logit_weight = (
            (head.output_weights if tied_logit_weight is None else tied_logit_weight).detach().requires_grad_()
        )
        normalization_weight = head.final_norm.weight.detach().requires_grad_()
        input_ = input_.detach().requires_grad_()

        hidden = torch.rms_norm(input_.to(normalization_weight.dtype), input_.shape[-1:], normalization_weight, 1e-5)
        logits = torch.nn.functional.linear(hidden, logit_weight).float()

        if self.final_logit_softcap is not None:
            cap = self.final_logit_softcap
            logits = torch.tanh(logits / cap) * cap

        if self.logits_scale_factor is not None:
            logits = logits * self.logits_scale_factor

        names_losses_weights = []

        loss_mask = (
            kwargs[LanguageModelKwargs.loss_mask][head._prediction_distance - 1]
            if LanguageModelKwargs.loss_mask in kwargs
            else None
        )

        if self.actual_label_loss is not False or self.grpo_loss is not False or self.gspo_loss is not False:
            labels = kwargs[LanguageModelKwargs.labels][head._prediction_distance - 1]

        if self.actual_label_loss is not False:
            label_loss = torch.nn.functional.cross_entropy(logits, labels)
            names_losses_weights.append(("label", label_loss, float(self.actual_label_loss)))

        if self.distillation_loss is not False:
            # Teacher logits are scaled by `logits_scale_factor / temperature` before the softmax, matching the kernel.
            teacher_logits = kwargs[f"reference_distillation_hidden_states"]["head.logits"].float()
            teacher_logits = teacher_logits * (self.logits_scale_factor / self.distillation_temperature)
            distillation_loss = torch.nn.functional.cross_entropy(
                logits,
                torch.softmax(teacher_logits, -1),
                reduction="mean" if loss_mask is None else "none",
            )
            if loss_mask is not None:
                distillation_loss = (distillation_loss * loss_mask).sum() / loss_mask.sum()
            names_losses_weights.append(("distillation", distillation_loss, float(self.distillation_loss)))

        if self.z_loss is not False:
            z_loss = torch.logsumexp(logits, dim=-1) ** 2
            if loss_mask is not None:
                z_loss = z_loss * loss_mask
            z_loss = z_loss.mean() if loss_mask is None else (z_loss * loss_mask).sum() / loss_mask.sum()
            names_losses_weights.append(("z_loss", z_loss, float(self.z_loss)))

        if self.grpo_loss is not False:
            grpo_loss, new_logprobs = reference_grpo_loss(
                logits,
                labels,
                kwargs[LanguageModelLossKwargs.advantages][head._prediction_distance - 1],
                kwargs[LanguageModelLossKwargs.old_log_probabilities][head._prediction_distance - 1],
            )
            names_losses_weights.append(("grpo_loss", grpo_loss, float(self.grpo_loss)))
            names_losses_weights.append(("grpo_loss_new_logprobs", new_logprobs, 0.0))

            if self.grpo_metrics is not None:
                # `logits` is already scaled above, so pass logits_scale_factor=1.0.
                metrics = reference_grpo_metrics(
                    logits,
                    labels,
                    kwargs[LanguageModelLossKwargs.advantages][head._prediction_distance - 1],
                    kwargs[LanguageModelLossKwargs.old_log_probabilities][head._prediction_distance - 1],
                    kwargs[LanguageModelLossKwargs.label_counts][head._prediction_distance - 1],
                    epsilon_low=0.2,
                    epsilon_high=0.2,
                    logits_scale_factor=1.0,
                    compute_entropy=self.grpo_metrics == "with_entropy",
                )
                num_documents = kwargs[LanguageModelKwargs.num_documents_in_batch]
                for attr in ("old_logprobs", "ratio_new_old", "kl_new_old", "clipped_ratio_fraction", "advantage"):
                    names_losses_weights.append((f"grpo_loss_{attr}", getattr(metrics, attr) / num_documents, 0.0))
                for attr in ("ratio_new_old_sum", "ratio_new_old_squared_sum", "num_tokens"):
                    names_losses_weights.append((f"grpo_loss_{attr}", getattr(metrics, attr), 0.0))
                names_losses_weights.append(("grpo_loss_max_advantage", metrics.max_advantage, 0.0))
                names_losses_weights.append(("grpo_loss_min_advantage", metrics.min_advantage, 0.0))
                if metrics.entropy is not None:
                    names_losses_weights.append(("grpo_loss_entropy", metrics.entropy / num_documents, 0.0))

        if self.gspo_loss is not False:
            gspo_loss, _ = reference_gspo_loss(
                logits,
                labels,
                kwargs[LanguageModelLossKwargs.advantages][head._prediction_distance - 1],
                kwargs[LanguageModelLossKwargs.old_log_probabilities][head._prediction_distance - 1],
                # `global_document_index_q` is 1-based per the data preprocessor convention; the reference takes 0-based.
                kwargs[BlockKwargs.global_document_index_q].long() - 1,
                kwargs[BlockKwargs.num_documents_in_sequence],
            )
            # Average over documents of per-document mean log-prob — matches the kernel's
            # `sum_t logprob_t * mask_t / label_count_t` divided by `num_documents_in_batch`.
            target_log_probabilities = (
                torch.nn.functional.log_softmax(logits, -1)
                .gather(-1, (labels * (labels >= 0)).unsqueeze(-1))
                .squeeze(-1)
            )
            label_mask = (labels >= 0).to(target_log_probabilities.dtype)
            document_means = [
                (document_log_probabilities * document_label_mask).sum() / document_label_mask.sum().clamp(min=1)
                for document_log_probabilities, document_label_mask in zip(
                    torch.split(target_log_probabilities, self.actual_gspo_document_lengths),
                    torch.split(label_mask, self.actual_gspo_document_lengths),
                    strict=True,
                )
            ]
            new_logprobs = torch.stack(document_means).mean()
            names_losses_weights.append(("gspo_loss", gspo_loss, float(self.gspo_loss)))
            names_losses_weights.append(("gspo_loss_new_logprobs", new_logprobs, 0.0))

        actual_losses = [loss * weight for _, loss, weight in names_losses_weights if weight != 0.0]
        total_loss = sum(actual_losses)
        total_loss.backward()
        losses = {LM_HEAD_LOSS_NAME: total_loss.detach()} | {
            name: loss.detach()
            for name, loss, weight in names_losses_weights
            if weight != 1.0 or len(actual_losses) > 1
        }

        if head._prediction_distance > 1:
            losses = {f"{name}_{head._prediction_distance}": loss for name, loss in losses.items()}

        return total_loss.detach(), input_.grad, logit_weight.grad, normalization_weight.grad, losses


_lm_head_test_configs = []


def _add_configs(base_name: str, **kwargs):
    # Loss masking and splits are important and error-prone, so we test them for all scenarios.
    for loss_masking in (False, True):
        for num_splits in (1, 2):
            _lm_head_test_configs.append(
                LMHeadTestConfig(
                    f"{base_name}{"_masked" if loss_masking else ""}{"" if num_splits == 1 else "_split"}",
                    loss_masking=loss_masking,
                    num_splits=num_splits,
                    **kwargs,
                )
            )


_add_configs("default")
_add_configs("bfloat16", compute_dtype=DataType.bfloat16)
_add_configs("full_precision_residual", full_precision_residual=True)
_add_configs("logit_scaling", logits_scale_factor=5.0)
_add_configs("final_logit_softcap", final_logit_softcap=2.0)
# Locks the softcap → scale ordering: scale * tanh(linear / cap) * cap, applied before softmax.
_add_configs("softcap_and_logit_scaling", final_logit_softcap=2.0, logits_scale_factor=5.0)
_add_configs("tied_embedding_weight", tied_embedding_weight=True)
_add_configs("multi_token_prediction", prediction_heads=2)
_add_configs("label_loss", label_loss=True)
_add_configs("distillation_loss", distillation_loss=True)
_add_configs("z_loss", z_loss=True)
_add_configs("grpo_loss", grpo_loss=True)
# GSPO can't be split: per-segment aggregation can't recombine across cross_entropy_splits chunks.
for loss_masking in (False, True):
    _lm_head_test_configs.append(
        LMHeadTestConfig(
            f"gspo_loss{'_masked' if loss_masking else ''}",
            gspo_loss=True,
            loss_masking=loss_masking,
        )
    )
_lm_head_test_configs.append(
    LMHeadTestConfig(
        "gspo_loss_uneven_documents",
        gspo_loss=True,
        gspo_document_lengths=(17, 31, 58, 94),
    )
)
_add_configs("label_and_distillation_loss", label_loss=True, distillation_loss=True)
_add_configs("label_and_z_loss_weighted", label_loss=True, z_loss=0.5)
_add_configs("label_and_distillation_loss_zero_weight", label_loss=True, distillation_loss=0.0)
_add_configs("distillation_loss_temperature", distillation_loss=True, distillation_temperature=2.0)

# Monolithic ("fused") head-loss path. Cross-entropy, z-loss, and the from-distribution (distillation) losses
# run in the monolithic kernel; unsupported losses fall back to their own implementation and accumulate into
# the same gradient.
_add_configs("fused", loss_implementation="fused")
_add_configs("fused_bfloat16", loss_implementation="fused", compute_dtype=DataType.bfloat16)
_add_configs("fused_logit_scaling", loss_implementation="fused", logits_scale_factor=5.0)
_add_configs("fused_final_logit_softcap", loss_implementation="fused", final_logit_softcap=2.0)
_add_configs("fused_tied_embedding_weight", loss_implementation="fused", tied_embedding_weight=True)
_add_configs("fused_multi_token_prediction", loss_implementation="fused", prediction_heads=2)
_add_configs("fused_label_and_z_loss_weighted", loss_implementation="fused", label_loss=True, z_loss=0.5)
_add_configs("fused_distillation_loss", loss_implementation="fused", distillation_loss=True)
_add_configs("fused_label_and_distillation_loss", loss_implementation="fused", label_loss=True, distillation_loss=True)
_add_configs(
    "fused_distillation_loss_temperature",
    loss_implementation="fused",
    distillation_loss=True,
    distillation_temperature=2.0,
)
_add_configs("fused_grpo_loss", loss_implementation="fused", grpo_loss=True)
# GSPO runs through its own three-phase path (compiled forward → eager segment seam → compiled backward),
# accumulating into the monolithic kernel's shared gradient. No splits (per-segment aggregation can't
# recombine across cross_entropy_splits chunks).
for loss_masking in (False, True):
    _lm_head_test_configs.append(
        LMHeadTestConfig(
            f"fused_gspo_loss{'_masked' if loss_masking else ''}",
            gspo_loss=True,
            loss_masking=loss_masking,
            loss_implementation="fused",
        )
    )
# GRPO metric family. Single-split only: per-split metric partials reduce across splits, which the
# whole-sequence reference doesn't model.
for _loss_implementation in ("per_loss", "fused"):
    _prefix = "" if _loss_implementation == "per_loss" else "fused_"
    for _metrics in ("basic", "with_entropy"):
        _suffix = "metrics" if _metrics == "basic" else "entropy"
        for _loss_masking in (False, True):
            _lm_head_test_configs.append(
                LMHeadTestConfig(
                    f"{_prefix}grpo_loss_{_suffix}{'_masked' if _loss_masking else ''}",
                    grpo_loss=True,
                    grpo_metrics=_metrics,
                    loss_masking=_loss_masking,
                    loss_implementation=_loss_implementation,
                )
            )


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(_lm_head_test_config, id=_lm_head_test_config.name)
        for _lm_head_test_config in _lm_head_test_configs
    ],
)
def test_lm_head(test_config: LMHeadTestConfig):
    model_config = test_config.get_config()
    model, distributed = get_base_model(model_config)
    input_, kwargs = test_config.get_inputs()

    tied_logit_weight = (
        torch.nn.Parameter(
            torch.empty(
                VOCAB_SIZE, HIDDEN_SIZE, dtype=distributed.config.compute_dtype.torch, device=distributed.device
            ).normal_(HIDDEN_SIZE**-0.5)
        )
        if test_config.tied_embedding_weight or test_config.prediction_heads > 1
        else None
    )

    for prediction_distance in range(1, model_config.base_model.head.prediction_heads + 1):
        # Prepare the LM head
        head: LanguageModelHead = (
            model.head if prediction_distance == 1 else model.multi_token_prediction.heads[prediction_distance - 2]
        )
        Assert.custom(isinstance, head, LanguageModelHead)
        Assert.eq(head._prediction_distance, prediction_distance)
        is_duplicate = test_config.tied_embedding_weight or prediction_distance > 1
        stage = get_stage(
            [head],
            distributed,
            tied_parameter_duplicates=[head.output_weights.tensor_name] if is_duplicate else [],
            tied_parameter_duplicate_buffers=(
                {head.output_weights.tensor_name: tied_logit_weight} if is_duplicate else {}
            ),
            # Names must be kept as-is for tied weights.
            set_names=False,
        )

        ref_total_loss, ref_input_grad, ref_logit_weight_grad, ref_normalization_weight_grad, ref_losses = (
            test_config.get_reference_outputs(
                head, input_, kwargs, tied_logit_weight if prediction_distance > 1 else None
            )
        )

        # Prepare LM head inputs
        if head._is_last_head:
            head_input = input_.detach().requires_grad_()
            output_grad = input_.new_full((), float("nan"))
        else:
            shared_hidden = torch.randn_like(input_)
            head_input = torch.stack((shared_hidden, input_.detach())).requires_grad_()
            output_grad = torch.randn_like(shared_hidden)

        if is_duplicate:
            logit_weight = tied_logit_weight
            logit_weight.grad_buffer = torch.full_like(logit_weight, float("nan"))
            logit_weight.param_grad_is_zero = True
        else:
            logit_weight = head.output_weights

        losses = collections.defaultdict(list)
        output, context = stage.forward(head_input, kwargs, losses)
        stage.backward(output_grad, context)
        threshold = 1e-5 if distributed.config.compute_dtype == DataType.float32 else 5e-3
        min_threshold = (
            1e-5 if distributed.config.compute_dtype == DataType.float32 else 1e-4
        ) * test_config.logits_scale_factor

        loss_definitions_ = head.get_loss_definitions()
        loss_definitions = {definition.name: definition for definition in loss_definitions_}
        Assert.eq(len(loss_definitions), len(loss_definitions_))
        Assert.eq(losses.keys(), ref_losses.keys(), loss_definitions.keys())

        losses = {
            name: loss_definition.reduce(losses[name], distributed)
            for name, loss_definition in loss_definitions.items()
        }

        for name, loss in losses.items():
            Assert.rms_close_relative(loss, ref_losses[name], threshold, min_threshold, msg=name)

        if head._is_last_head:
            # Assert.all_equal(output, losses[lm_head_loss_name][0])
            input_grad = head_input.grad
        else:
            Assert.all_equal(output, shared_hidden)
            shared_hidden_grad, input_grad = head_input.grad.unbind()
            Assert.all_equal(shared_hidden_grad, output_grad)

        Assert.rms_close_relative(input_grad, ref_input_grad, threshold, min_threshold)
        Assert.rms_close_relative(
            head.final_norm.weight.grad_buffer, ref_normalization_weight_grad, threshold, min_threshold
        )
        Assert.rms_close_relative(logit_weight.grad_buffer, ref_logit_weight_grad, threshold, min_threshold)
