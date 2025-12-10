import numpy as np
import pytest
import torch

from fast_llm.functional.config import ActivationType, MLPRecomputeLevel
from fast_llm.functional.dpo import compute_dpo_loss
from fast_llm.functional.triton.mlp import mlp_autograd, mlp_autograd_looped, torch_mlp_activation
from fast_llm.functional.triton.sparse_copy import get_sparse_map
from fast_llm.utils import Assert
from tests.utils.dataset import get_random_spans
from tests.utils.utils import requires_cuda


def _get_target_log_probability_for_spans(log_probabilities: torch.Tensor, spans: list[list[tuple[int, int]]]):
    return sum(
        log_probabilities[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(spans)
        for begin, end in sample_spans
    )


def reference_dpo_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reference_model_logits: torch.Tensor,
    chosen_spans: torch.Tensor,
    rejected_spans: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    # TODO: Too similar to the actual implementation.
    policy_log_probs = (
        torch.nn.functional.log_softmax(logits.float(), dim=-1).gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    )
    policy_chosen_logps = sum(
        policy_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(chosen_spans)
        for begin, end in sample_spans
    )
    policy_rejected_logps = sum(
        policy_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(rejected_spans)
        for begin, end in sample_spans
    )
    reference_log_probs = (
        torch.nn.functional.log_softmax(reference_model_logits.float(), dim=-1)
        .gather(dim=-1, index=targets.unsqueeze(-1))
        .squeeze(-1)
    )
    reference_chosen_logps = sum(
        reference_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(chosen_spans)
        for begin, end in sample_spans
    )
    reference_rejected_logps = sum(
        reference_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(rejected_spans)
        for begin, end in sample_spans
    )
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    return -torch.nn.functional.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()


def test_dpo_loss():
    random_state = np.random.RandomState(0)
    logits = torch.from_numpy(random_state.normal(size=(10, 50, 100))).to(torch.float32).requires_grad_()
    reference_model_logits = torch.from_numpy(random_state.normal(size=(10, 50, 100))).to(torch.float32)
    targets = torch.from_numpy(random_state.randint(0, 100, (10, 50)))

    spans = get_random_spans(np.full(10, 50), 0, 10, random_state)

    fastllm_loss, fast_llm_grad = compute_dpo_loss(
        logits, targets, reference_model_logits, spans[::2], spans[1::2], beta=1, grad_output=1
    )
    reference_loss = reference_dpo_loss(logits, targets, reference_model_logits, spans[::2], spans[1::2], beta=1)
    reference_loss.backward()
    Assert.rms_close(fastllm_loss, reference_loss, 1e-5)
    Assert.rms_close(fast_llm_grad, logits.grad, 1e-5)


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
@pytest.mark.skip("Dropless MoE is broken")
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
