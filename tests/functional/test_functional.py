import random

import pytest
import torch

from fast_llm.functional.config import ActivationType, MLPRecomputeLevel
from fast_llm.functional.dpo import _compute_dpo_loss, _compute_logprobs_for_preference_spans
from fast_llm.functional.triton.mlp import mlp_autograd, mlp_autograd_looped, torch_mlp_activation
from fast_llm.functional.triton.sparse_copy import get_sparse_map
from fast_llm.utils import Assert
from tests.utils.utils import requires_cuda


def ref_log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        logits.div_(temperature)
    batch_dim = logits.shape[:-1]
    last_dim = logits.shape[-1]

    output = torch.nn.functional.cross_entropy(logits.reshape(-1, last_dim), labels.reshape(-1), reduction="none")
    log_probs_labels = -output.view(*batch_dim)

    return log_probs_labels


def ref_packed_get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    attention_mask,
    prompt_id_lens,
    packed_seq_lens,
) -> torch.FloatTensor:
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]
    per_token_logps = ref_log_probs_from_logits(logits, labels)

    loss_masks = attention_mask.clone().bool()

    index = 0
    for i, seq_len in enumerate(packed_seq_lens):
        loss_masks[0, index : index + prompt_id_lens[i]] = False
        index = index + seq_len

    loss_masks = loss_masks[:, 1:]

    logprobs_sums = []
    index = 0
    for i, seq_len in enumerate(packed_seq_lens):
        seq = per_token_logps[0, index : index + seq_len - 1]
        mask = loss_masks[0, index : index + seq_len - 1]
        logprobs_sums.append((seq * mask).sum())
        index = index + seq_len
    chosen_logps = logprobs_sums[: len(packed_seq_lens) // 2]
    rejected_logps = logprobs_sums[len(packed_seq_lens) // 2 :]

    return torch.tensor(chosen_logps), torch.tensor(rejected_logps)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("batch_size", "seq_length", "vocab_size"),
    (
        (2, 32, 50),
        (1, 32, 50),
        (2, 100, 50),
        (2, 32, 200),
    ),
)
def test_preference_logps(batch_size, seq_length, vocab_size):
    random.seed(0)
    torch.manual_seed(0)

    def random_split(seq_length):
        min_val = int(seq_length * 0.3)
        max_val = int(seq_length * 0.7)

        if max_val < min_val:
            max_val = min_val

        a = random.randint(min_val, max_val)
        b = seq_length - a
        return [a, b]

    logits = torch.randn(batch_size, seq_length, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))
    packed_seq_lens = random_split(seq_length)  # simulate different chosen/rejected lengths
    prompt_id_lens = [int(min(packed_seq_lens) * 0.75)] * 2  # sequences are 75% prompt 25% generation
    attention_mask = torch.tensor([1] * packed_seq_lens[0] + [2] * packed_seq_lens[1]).unsqueeze(0)

    chosen_span = torch.tensor([[prompt_id_lens[0], packed_seq_lens[0] - 1]]) - 1  # shift by 1 due to label shifting
    rejected_span = (
        torch.tensor([[packed_seq_lens[0] + prompt_id_lens[1], packed_seq_lens[0] + packed_seq_lens[1] - 1]]) - 1
    )  # shift by 1 due to label shifting

    ref_chosen_logps, ref_rejected_logps = ref_packed_get_batch_logps(
        logits, targets, attention_mask, prompt_id_lens, packed_seq_lens
    )

    chosen_logps, rejected_logps, selected_log_probs = _compute_logprobs_for_preference_spans(
        logits=logits,
        targets=targets[:, 1:],
        chosen_spans=chosen_span,
        rejected_spans=rejected_span,
    )

    ref_logps = ref_log_probs_from_logits(logits[:, :-1, :], targets[:, 1:])

    # check all logps
    Assert.custom(torch.allclose, ref_logps, selected_log_probs, rtol=1e-5)

    # check chosen and rejected summed logps
    Assert.custom(torch.allclose, ref_chosen_logps, chosen_logps, rtol=1e-5)
    Assert.custom(torch.allclose, ref_rejected_logps, rejected_logps, rtol=1e-5)


def ref_dpo_loss_fcn(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta=1,
    label_smoothing=0,
) -> torch.Tensor:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = (
        -torch.nn.functional.logsigmoid(beta * logits) * (1 - label_smoothing)
        - torch.nn.functional.logsigmoid(-beta * logits) * label_smoothing
    )

    loss = losses.mean()

    return loss


def test_dpo_loss():
    torch.manual_seed(0)

    NUM_SAMPLES = 20
    policy_chosen_logps = torch.rand(NUM_SAMPLES)
    policy_rejected_logps = torch.rand(NUM_SAMPLES)
    reference_chosen_logps = torch.rand(NUM_SAMPLES)
    reference_rejected_logps = torch.rand(NUM_SAMPLES)
    betas = torch.rand(NUM_SAMPLES)

    for i in range(NUM_SAMPLES):
        fastllm_dpo_loss = _compute_dpo_loss(
            policy_chosen_logps=policy_chosen_logps[i],
            policy_rejected_logps=policy_rejected_logps[i],
            reference_chosen_logps=reference_chosen_logps[i],
            reference_rejected_logps=reference_rejected_logps[i],
            beta=betas[i].item(),
        )
        ref_dpo_loss = ref_dpo_loss_fcn(
            policy_chosen_logps=policy_chosen_logps[i].unsqueeze(0),
            policy_rejected_logps=policy_rejected_logps[i].unsqueeze(0),
            reference_chosen_logps=reference_chosen_logps[i].unsqueeze(0),
            reference_rejected_logps=reference_rejected_logps[i].unsqueeze(0),
            beta=betas[i].item(),
        )
        Assert.rms_close(fastllm_dpo_loss, ref_dpo_loss, 1e-5)


@requires_cuda
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize(
    "activation", [ActivationType.gelu, ActivationType.silu, ActivationType.relu, ActivationType.squared_relu]
)
def test_mlp_recomputation(gated, activation):
    tokens = 1024
    hidden_size = 2048
    intermediate_size = 4096
    std = 1 / 64
    input_ = torch.randn(tokens, hidden_size, device="cuda", requires_grad=True)
    output_grad = torch.randn(tokens, hidden_size, device="cuda", requires_grad=True)
    weight_1 = torch.normal(0, std, (intermediate_size * (gated + 1), hidden_size), device="cuda", requires_grad=True)
    bias_1 = torch.normal(0, std, (intermediate_size * (gated + 1),), device="cuda", requires_grad=True)
    weight_2 = torch.normal(0, std, (intermediate_size, hidden_size), device="cuda", requires_grad=True)
    bias_2 = torch.normal(0, std, (hidden_size,), device="cuda", requires_grad=True)
    params = (weight_1, bias_1, weight_2, bias_2)

    output_ref = torch.nn.functional.linear(
        torch_mlp_activation(
            (torch.nn.functional.linear(input_, weight_1, bias_1)),
            gated,
            activation,
        ),
        weight_2.t(),
        bias_2,
    )
    output_ref.backward(output_grad)
    input_grad_ref = input_.grad
    param_grad_refs = [param.grad for param in params]

    for i, recompute_level in enumerate(MLPRecomputeLevel):
        print(recompute_level.value)  # noqa
        input_.grad = None
        for param in params:
            param.grad = None
            param.grad_buffer = torch.empty_like(param)
            param.param_grad_is_zero = True
        output = mlp_autograd(input_, None, *params, gated, activation, None, False, True, recompute_level, True)
        output.backward(output_grad)
        if i == 0:
            Assert.rms_close(output, output_ref, 1e-5)
            Assert.rms_close(input_.grad, input_grad_ref, 1e-5)
            for param, param_grad_ref in zip(params, param_grad_refs):
                Assert.rms_close(param.grad_buffer, param_grad_ref, 3e-4)
            output_ref = output
            input_grad_ref = input_.grad
            param_grad_refs = [param.grad_buffer for param in params]
        else:
            # Recomputation doesn't modify the output.
            Assert.all_equal(output, output_ref)
            Assert.all_equal(input_.grad, input_grad_ref)
            for param, param_grad_ref in zip(params, param_grad_refs):
                Assert.all_equal(param.grad_buffer, param_grad_ref)


# Takes ~6s, much more if it needs to compile, reducing the hidden size doesn't help.
@pytest.mark.slow
@requires_cuda
def test_dropless_mlp():
    num_experts = 4
    experts_per_token = 4
    tokens = 256
    hidden_size = 512
    intermediate_size = 1024
    std = 1 / 64
    input_ = torch.randn(tokens, hidden_size, device="cuda", requires_grad=True)
    router_weight = torch.normal(0, std, (num_experts, hidden_size), device="cuda")

    top_logits, top_experts = torch.topk(
        torch.nn.functional.linear(input_.detach(), router_weight), k=experts_per_token, dim=-1
    )
    scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).detach().requires_grad_()

    output_grad = torch.randn(tokens, hidden_size, device="cuda", requires_grad=True)
    weight_1 = torch.normal(
        0, std, (intermediate_size * 2 * num_experts, hidden_size), device="cuda", requires_grad=True
    )
    weight_2 = torch.normal(0, std, (intermediate_size * num_experts, hidden_size), device="cuda", requires_grad=True)
    params = (weight_1, weight_2)

    for param in params:
        param.grad = None
        param.grad_buffer = torch.empty_like(param)
        param.param_grad_is_zero = True
    output_ref = mlp_autograd_looped(
        input_,
        scores,
        top_experts,
        weight_1,
        weight_2,
        num_experts,
        True,
        ActivationType.silu,
        None,
        False,
        True,
        MLPRecomputeLevel.none,
    )
    output_ref.backward(output_grad)
    input_grad_ref = input_.grad
    scores_grad_ref = scores.grad
    param_grad_refs = [param.grad_buffer for param in params]

    sparse_map = get_sparse_map(top_experts, num_experts)

    for i, recompute_level in enumerate(MLPRecomputeLevel):
        print("recompute_level", recompute_level)  # noqa
        input_.grad = None
        scores.grad = None
        for param in params:
            param.grad = None
            param.grad_buffer = torch.empty_like(param)
            param.param_grad_is_zero = True
        output = mlp_autograd(
            input_,
            scores,
            weight_1,
            None,
            weight_2,
            None,
            True,
            ActivationType.silu,
            None,
            False,
            True,
            recompute_level,
            True,
            sparse_map,
        )

        output.backward(output_grad)
        if i == 0:
            # TODO: Thresholds are a bit high.
            Assert.rms_close(output, output_ref, 1e-3)
            Assert.rms_close(input_.grad, input_grad_ref, 1e-3)
            Assert.rms_close(scores.grad, scores_grad_ref, 5e-2)
            for param, param_grad_ref in zip(params, param_grad_refs):
                Assert.rms_close(param.grad_buffer, param_grad_ref, 1e-2)
            output_ref = output
            input_grad_ref = input_.grad
            scores_grad_ref = scores.grad
            param_grad_refs = [param.grad_buffer for param in params]
        else:
            # Recomputation doesn't modify the output.
            Assert.all_equal(output, output_ref)
            Assert.all_equal(input_.grad, input_grad_ref)
            Assert.all_equal(scores.grad, scores_grad_ref)
            for param, param_grad_ref in zip(params, param_grad_refs):
                Assert.all_equal(param.grad_buffer, param_grad_ref)
