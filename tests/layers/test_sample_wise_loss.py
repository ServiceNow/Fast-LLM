"""Independent packed-document references for sample-wise loss normalization."""

from types import SimpleNamespace

import pytest
import torch

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument, LanguageModelInput
from fast_llm.data.document.range import RangeDocument
from fast_llm.engine.distributed.config import DistributedBackend, DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.functional.config import EntropyLossType, TargetFormat
from fast_llm.functional.entropy_loss import fused_entropy_loss_forward_backward
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.loss.config import LanguageModelLabelEntropyLossConfig


def _documents(index: int, all_masked: bool = False) -> list[LanguageModelDocument]:
    lengths = [2, 161, 35, 23]
    masks = [[(0, 0)], [(0, 13), (38, 49), (77, 89)], [(0, 35)], [(0, 4)]]
    if index % 2:
        masks[-1] = [(0, 23)]
    if all_masked:
        masks = [[(0, length)] for length in lengths]
    return [
        LanguageModelDocument(
            tokens=(torch.arange(length) + index + offset) % 19,
            loss_masking_spans=RangeDocument(ranges=spans),
        )
        for offset, (length, spans) in enumerate(zip(lengths, masks, strict=True))
    ]


def _reference(logits, documents, distance, reduction="sample"):
    losses = []
    document_means = []
    counts = torch.zeros(258, dtype=torch.int64, device=logits.device)
    offset = 0
    for document in documents:
        positions = [
            position
            for position in range(distance, len(document.tokens))
            if not any(begin <= position < end for begin, end in document.loss_masking_spans.ranges)
        ]
        counts[offset : offset + len(document.tokens)] = len(positions)
        if positions:
            row_losses = torch.nn.functional.cross_entropy(
                logits[[offset + position - distance for position in positions]].float() * 0.7,
                document.tokens[positions].to(logits.device),
                reduction="none",
            )
            losses.extend(row_losses.unbind())
            document_means.append(row_losses.mean())
        offset += len(document.tokens)
    terms = document_means if reduction == "sample" else losses
    return terms, counts


def _mean_or_zero(terms, logits):
    return torch.stack(terms).mean() if terms else logits.float().sum() * 0


@pytest.mark.parametrize("reduction", ["token", "sample"])
@pytest.mark.parametrize("all_masked", [False, True])
@pytest.mark.parametrize("use_triton", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_sample_end_to_end(reduction, all_masked, use_triton, dtype):
    if not torch.cuda.is_available():
        pytest.skip("GPU backend integration")
    device = torch.device("cuda")
    distributed = SimpleNamespace(batch_data_group=None)
    config = DistributedConfig(use_cuda=True)
    loss_config = LanguageModelLabelEntropyLossConfig(reduction=reduction, use_triton=use_triton, weight=0.8)
    preprocessing = LanguageModelBatchPreprocessingConfig(
        distributed=config,
        predicted_tokens=2,
        micro_batch_splits=2,
        **loss_config.get_layer(config, name="ce").get_preprocessing_config(),
    )
    documents = _documents(0, all_masked)
    batch = LanguageModelBatch.from_documents(documents, pad_to_size=258).to_device_(device)
    inputs = batch.get_model_inputs(preprocessing)
    LanguageModelInput.share_batch_data(inputs, distributed)
    for distance in [1, 2]:
        logits = torch.randn(256, 19, device=device, dtype=dtype, requires_grad=True)
        terms, counts = _reference(logits, documents, distance, reduction)
        expected = _mean_or_zero(terms, logits) * 0.8
        expected_grad = torch.autograd.grad(expected * 1.3, logits)[0]
        layer = loss_config.get_layer(
            config, name="ce", prediction_distance=distance, prediction_heads=2, num_splits=8, logits_scale_factor=0.7
        )
        loss = torch.zeros((), device=device)
        gradients = []
        for index, model_input in enumerate(inputs):
            kwargs = model_input.to_kwargs()
            kwargs[LanguageModelKwargs.grad_output] = 1.3
            if reduction == "sample":
                target = model_input.targets[distance - 1]
                assert target.num_valid_documents_in_batch == len(terms)
                begin = index * 128 + distance
                torch.testing.assert_close(target.label_counts, counts[begin : begin + 128])
            for split, chunk in enumerate(logits.detach()[index * 128 : (index + 1) * 128].chunk(8)):
                initial = torch.full_like(chunk, 0.25)
                # Token mode's existing zero-label divisor semantics are outside this feature.
                if reduction == "token" and all_masked:
                    kwargs[LanguageModelKwargs.num_labels_in_batch] = [1, 1]
                actual, grad = layer.forward_backward(
                    chunk.contiguous(), kwargs, split_index=split, grad_logits=initial.clone()
                )
                loss += actual
                gradients.append(grad)
        torch.testing.assert_close(loss, expected.detach(), rtol=2e-5, atol=2e-5)
        # Production casts the new contribution before adding to the existing buffer.
        expected_accumulated = torch.full_like(logits, 0.25).add(expected_grad)
        torch.testing.assert_close(
            torch.cat(gradients),
            expected_accumulated,
            rtol=0.02 if dtype == torch.bfloat16 else 2e-5,
            atol=0.002 if dtype == torch.bfloat16 else 2e-6,
        )


def _run_sample_parallel(context, base_path, sequence_data_parallel, all_masked=False):
    with context.subtest(base_path, f"sequence_data_{sequence_data_parallel}_masked_{all_masked}", 8) as subtest:
        if not subtest.do_run:
            return
        config = DistributedConfig(
            tensor_parallel=2,
            sequence_tensor_parallel=True,
            sequence_data_parallel=sequence_data_parallel,
            backend=DistributedBackend.nccl,
        )
        distributed = Distributed(config)
        device = distributed.device
        loss_config = LanguageModelLabelEntropyLossConfig(reduction="sample", weight=0.8)
        preprocessing = LanguageModelBatchPreprocessingConfig(
            distributed=config,
            predicted_tokens=2,
            micro_batch_splits=2,
            return_label_counts=True,
            return_valid_document_count=True,
        )
        # Two accumulated packed batches per batch-data peer, with unequal document counts.
        global_documents = [_documents(index, all_masked) for index in range(2 * config.batch_data_parallel)]
        ids = [2 * config.batch_data_rank, 2 * config.batch_data_rank + 1]
        inputs_by_id = {
            index: LanguageModelBatch.from_documents(global_documents[index], pad_to_size=258)
            .to_device_(device)
            .get_model_inputs(preprocessing)
            for index in ids
        }
        LanguageModelInput.share_batch_data([item for inputs in inputs_by_id.values() for item in inputs], distributed)
        generator = torch.Generator(device=device).manual_seed(174)
        features = [torch.randn(256, 5, generator=generator, device=device) for _ in global_documents]
        parameter = torch.randn(5, 19, generator=generator, device=device, requires_grad=True)
        logits = [feature @ parameter for feature in features]
        for distance in [1, 2]:
            terms = []
            expected_counts = []
            for scores, documents in zip(logits, global_documents, strict=True):
                document_terms, counts = _reference(scores, documents, distance)
                terms.extend(document_terms)
                expected_counts.append(counts)
            expected = _mean_or_zero(terms, logits[0])
            expected_parameter_grad = torch.autograd.grad(expected * 0.8 * 1.3, parameter, retain_graph=True)[0]
            for use_triton in [False, True]:
                loss_config = LanguageModelLabelEntropyLossConfig(
                    reduction="sample", weight=0.8, use_triton=use_triton
                )
                layer = loss_config.get_layer(
                    config,
                    name="ce",
                    prediction_distance=distance,
                    prediction_heads=2,
                    num_splits=8,
                    logits_scale_factor=0.7,
                    register_loss=True,
                )
                assembled_loss = torch.zeros((), device=device)
                parameter_grad = torch.zeros_like(parameter)
                logged = {"ce": []}
                for index, inputs in inputs_by_id.items():
                    for micro_index, model_input in enumerate(inputs):
                        target = model_input.targets[distance - 1]
                        assert target.num_valid_documents_in_batch == len(terms)
                        sequence_begin = micro_index * 128 + config.sequence_data_rank * (
                            128 // sequence_data_parallel
                        )
                        length = model_input.token_dim.size
                        torch.testing.assert_close(
                            target.label_counts,
                            expected_counts[index][sequence_begin + distance : sequence_begin + distance + length],
                        )
                        tensor_begin = sequence_begin + config.tensor_rank * (length // 2)
                        local_features = features[index][tensor_begin : tensor_begin + length // 2]
                        local_logits = logits[index].detach()[tensor_begin : tensor_begin + length // 2]
                        kwargs = model_input.to_kwargs()
                        kwargs[LanguageModelKwargs.grad_output] = 1.3 * config.data_parallel
                        for split, (chunk, feature) in enumerate(
                            zip(local_logits.chunk(8), local_features.chunk(8), strict=True)
                        ):
                            loss, grad = layer.forward_backward(chunk.contiguous(), kwargs, logged, split_index=split)
                            assembled_loss += loss
                            parameter_grad += feature.T @ grad
                torch.distributed.all_reduce(assembled_loss, group=context.group)
                # Loss registration already sums and replicates each sequence-tensor contribution.
                assembled_loss /= config.tensor_parallel
                torch.distributed.all_reduce(parameter_grad, group=context.group)
                parameter_grad /= config.data_parallel
                logged_loss = sum(logged["ce"])
                torch.distributed.all_reduce(logged_loss, group=distributed.data_group)
                torch.testing.assert_close(assembled_loss, expected.detach() * 0.8, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(logged_loss, expected.detach(), rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(parameter_grad, expected_parameter_grad, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize(
    ("sequence_data_parallel", "all_masked"),
    [
        pytest.param(2, False, id="unequal_batch_peers"),
        pytest.param(4, False, id="smoke_topology"),
        pytest.param(4, True, id="all_masked"),
    ],
)
def test_sample_exact_parallel(run_parallel_script, result_path, sequence_data_parallel, all_masked):
    if torch.cuda.device_count() < 8:
        pytest.skip("Requires eight GPUs")
    run_parallel_script(
        _run_sample_parallel,
        (result_path / "sample_exact_parallel", sequence_data_parallel, all_masked),
        world_size=8,
        backend=DistributedBackend.nccl,
        use_cuda=True,
        timeout=300,
    )


@pytest.mark.parametrize("backend", [fused_entropy_loss_forward_backward, triton_entropy_loss_forward_backward])
@pytest.mark.parametrize("target_format", [TargetFormat.probabilities, TargetFormat.logits])
@pytest.mark.parametrize("loss_type", list(EntropyLossType))
def test_weighted_distribution_targets(backend, target_format, loss_type):
    if not torch.cuda.is_available():
        pytest.skip("GPU entropy backend")
    logits = torch.randn(7, 19, device="cuda", requires_grad=True)
    targets = torch.randn_like(logits)
    probabilities = targets.softmax(-1)
    target = probabilities if target_format == TargetFormat.probabilities else targets
    weights = torch.tensor([0.1, 0.4, 0, 0.2, 0.7, 0.3, 0.9], device="cuda")
    mask = torch.tensor([True, False, True, True, True, False, True], device="cuda")
    log_probability = logits.log_softmax(-1)
    if loss_type == EntropyLossType.cross_entropy:
        rows = -(probabilities * log_probability).sum(-1)
    elif loss_type == EntropyLossType.forward_kl:
        rows = (probabilities * (probabilities.log() - log_probability)).sum(-1)
    else:
        rows = (log_probability.exp() * (log_probability - probabilities.log())).sum(-1)
    expected = (rows * weights * mask).sum() / 3
    expected_grad = torch.autograd.grad(expected * 1.2, logits)[0]
    initial = torch.randn_like(logits)
    loss, grad = backend(
        logits.detach(),
        target,
        mask,
        grad_logits=initial.clone(),
        grad_output=1.2,
        target_format=target_format,
        entropy_loss_type=loss_type,
        divisor=3,
        weights=weights,
    )
    torch.testing.assert_close(loss, expected.detach(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(grad, expected_grad + initial, rtol=2e-5, atol=2e-6)


def test_sample_configuration_validation():
    from fast_llm.config import ValidationError

    config = LanguageModelLabelEntropyLossConfig.from_dict({"reduction": "sample"})
    assert LanguageModelLabelEntropyLossConfig.from_dict(config.to_dict()).reduction == config.reduction
    with pytest.raises((ValidationError, ValueError, TypeError)):
        LanguageModelLabelEntropyLossConfig.from_dict({"reduction": "invalid"})


@pytest.mark.parametrize("request_counts", [False, True])
def test_unused_document_counts_not_computed(monkeypatch, request_counts):
    calls = []
    original = LanguageModelBatch._get_label_counts

    def count_labels(self, mask, return_valid_document_count=False):
        calls.append(return_valid_document_count)
        return original(self, mask, return_valid_document_count)

    monkeypatch.setattr(LanguageModelBatch, "_get_label_counts", count_labels)
    config = LanguageModelBatchPreprocessingConfig(return_label_counts=request_counts)
    inputs = LanguageModelBatch.from_documents(_documents(0), pad_to_size=258).get_model_inputs(config)
    assert calls == ([False] if request_counts else [])
    assert inputs[0].targets[0].num_valid_documents is None
    assert LanguageModelKwargs.num_valid_documents_in_batch not in inputs[0].to_kwargs()
    assert (
        LanguageModelLabelEntropyLossConfig().get_layer(config.distributed, name="ce").get_preprocessing_config() == {}
    )


def test_meta_counts_no_value_extraction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Meta preprocessing must not extract tensor values")

    for name in ["item", "tolist", "cpu"]:
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    config = LanguageModelBatchPreprocessingConfig(
        predicted_tokens=2, micro_batch_splits=2, return_label_counts=True, return_valid_document_count=True
    )
    inputs = (
        LanguageModelBatch.from_documents(_documents(0), pad_to_size=258)
        .to_device_(torch.device("meta"))
        .get_model_inputs(config)
    )
    for distance in range(2):
        assert inputs[0].targets[distance].num_valid_documents == 4
        assert inputs[0].targets[distance].num_labels == 258
        assert inputs[1].targets[distance].num_valid_documents == 0
        assert inputs[1].targets[distance].num_labels == 0


def test_shared_distillation_keeps_token_divisor(monkeypatch):
    from fast_llm.layers.language_model.loss import entropy_loss
    from fast_llm.layers.language_model.loss.config import LanguageModelDistillationLossConfig

    config = DistributedConfig(use_cuda=False)
    layer = LanguageModelDistillationLossConfig(use_triton=False).get_layer(config, name="distillation")
    target = torch.randn(8, 19)
    monkeypatch.setattr(layer, "_get_reference_model_logits", lambda *args: target)
    received = []
    original = entropy_loss.fused_entropy_loss_forward_backward

    def capture(*args, **kwargs):
        received.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(entropy_loss, "fused_entropy_loss_forward_backward", capture)
    logits = torch.randn(8, 19)
    mask = torch.tensor([True, False, True, True, False, True, True, False])
    kwargs = {
        LanguageModelKwargs.num_labels_in_batch: [5],
        LanguageModelKwargs.num_valid_documents_in_batch: [2],
        LanguageModelKwargs.loss_mask: [mask],
        LanguageModelKwargs.grad_output: 1.0,
    }
    actual, grad = layer.forward_backward(logits, kwargs)
    reference_logits = logits.clone().requires_grad_()
    expected = (-(target.softmax(-1) * reference_logits.log_softmax(-1)).sum(-1) * mask).sum() / 5
    expected_grad = torch.autograd.grad(expected, reference_logits)[0]
    torch.testing.assert_close(actual, expected.detach())
    torch.testing.assert_close(grad, expected_grad)
    assert received[0]["divisor"] == 5
    assert received[0].get("weights") is None


def test_training_diagnostic_reports(tmp_path, monkeypatch):
    from fast_llm.layers.language_model.loss.entropy_loss import LanguageModelLabelEntropyLoss
    from tools.validate_sample_wise_training import install_checks

    monkeypatch.setattr(LanguageModelBatch, "_set_target_inputs", LanguageModelBatch._set_target_inputs)
    monkeypatch.setattr(
        LanguageModelLabelEntropyLoss, "_forward_backward", LanguageModelLabelEntropyLoss._forward_backward
    )
    install_checks(tmp_path, 1)
    config = DistributedConfig(use_cuda=False)
    preprocessing = LanguageModelBatchPreprocessingConfig(
        distributed=config, predicted_tokens=2, return_label_counts=True, return_valid_document_count=True
    )
    inputs = LanguageModelBatch.from_documents(_documents(0), pad_to_size=258).get_model_inputs(preprocessing)
    LanguageModelInput.share_batch_data(inputs, SimpleNamespace(batch_data_group=None))
    layer = LanguageModelLabelEntropyLossConfig(reduction="sample", use_triton=False).get_layer(config, name="ce")
    kwargs = inputs[0].to_kwargs()
    kwargs[LanguageModelKwargs.grad_output] = 1.0
    layer.forward_backward(torch.randn(256, 19), kwargs)
    report = torch.load(tmp_path / "rank_0_call_0.pt", weights_only=True)
    assert report["preprocessed_batches_checked"] == 1
    assert report["global_document_divisor"] == 3
    torch.testing.assert_close(report["actual_loss"], report["reference_loss"])
    layer.forward_backward(torch.randn(256, 19), kwargs)
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_sample_preprocessing_single_host_transfer(monkeypatch):
    documents = _documents(0)
    batch = LanguageModelBatch.from_documents(documents, pad_to_size=258)
    config = LanguageModelBatchPreprocessingConfig(
        predicted_tokens=2, micro_batch_splits=2, return_label_counts=True, return_valid_document_count=True
    )
    inputs = batch.get_model_inputs(config)
    for model_input in inputs:
        model_input.targets.clear()
    calls = []
    original_cpu = torch.Tensor.cpu

    def transfer(tensor, *args, **kwargs):
        calls.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    def forbidden_item(*args, **kwargs):
        raise AssertionError("Sample preprocessing must transfer both local counts together")

    monkeypatch.setattr(torch.Tensor, "cpu", transfer)
    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    batch._set_target_inputs(inputs, config)
    assert calls == [(2,), (2,)]


def _run_weighted_vocab_precision(context, base_path):
    with context.subtest(base_path, "weighted_vocab_precision", 2) as subtest:
        if not subtest.do_run:
            return
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(92)
        for dtype in [torch.float32, torch.bfloat16]:
            full_logits = torch.randn(9, 38, generator=generator, device=device).to(dtype).requires_grad_()
            target_logits = torch.randn(9, 38, generator=generator, device=device).to(dtype)
            labels = torch.tensor([1, -100, 30, 4, 5, -100, 7, 8, -100], device=device)
            weights = torch.tensor([0.5, 0, 0.5, 0.25, 0.25, 0, 0.25, 0.25, 0], device=device)
            predicted_log_probability = full_logits.float().log_softmax(-1)
            probabilities = target_logits.float().softmax(-1)
            for target_format in TargetFormat:
                for loss_type in EntropyLossType:
                    if target_format == TargetFormat.labels:
                        if loss_type == EntropyLossType.reverse_kl:
                            continue
                        target = labels
                        rows = torch.nn.functional.cross_entropy(full_logits.float(), labels, reduction="none")
                    else:
                        target = target_logits if target_format == TargetFormat.logits else probabilities
                        if loss_type == EntropyLossType.cross_entropy:
                            rows = -(probabilities * predicted_log_probability).sum(-1)
                        elif loss_type == EntropyLossType.forward_kl:
                            rows = (probabilities * (probabilities.log() - predicted_log_probability)).sum(-1)
                        else:
                            rows = (
                                predicted_log_probability.exp() * (predicted_log_probability - probabilities.log())
                            ).sum(-1)
                    expected = (rows * weights).sum() / 2
                    expected_grad = torch.autograd.grad(expected * 1.3, full_logits, retain_graph=True)[0]
                    local_logits = full_logits.detach().chunk(2, -1)[context.rank].contiguous()
                    local_target = (
                        target
                        if target_format == TargetFormat.labels
                        else target.chunk(2, -1)[context.rank].contiguous()
                    )
                    initial = torch.full_like(local_logits, 0.01)
                    expected_accumulated = initial + expected_grad.chunk(2, -1)[context.rank]
                    for backend in [fused_entropy_loss_forward_backward, triton_entropy_loss_forward_backward]:
                        loss, grad = backend(
                            local_logits,
                            local_target,
                            None,
                            grad_logits=initial.clone(),
                            grad_output=1.3,
                            group=context.group,
                            target_format=target_format,
                            entropy_loss_type=loss_type,
                            divisor=2,
                            weights=weights,
                        )
                        torch.testing.assert_close(loss, expected.detach(), rtol=3e-5, atol=3e-6)
                        torch.testing.assert_close(
                            grad,
                            expected_accumulated,
                            rtol=0.02 if dtype == torch.bfloat16 else 3e-5,
                            atol=0.0002 if dtype == torch.bfloat16 else 3e-6,
                        )


def test_weighted_vocab_precision(run_parallel_script, result_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    run_parallel_script(
        _run_weighted_vocab_precision,
        (result_path / "weighted_vocab_precision",),
        world_size=2,
        backend=DistributedBackend.nccl,
        use_cuda=True,
    )
