"""Opt-in numerical checks for early non-vocabulary-parallel sample-loss calls.

Launch with the ordinary train CLI arguments, plus --debug-dir and --debug-splits.
This intentionally adds synchronization and reference computation during diagnostics.
"""

import argparse
import os
import pathlib

import torch

from fast_llm.cli import fast_llm_main
from fast_llm.data.document.language_model import LanguageModelBatch
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.loss.config import LanguageModelLabelLossReduction
from fast_llm.layers.language_model.loss.entropy_loss import LanguageModelLabelEntropyLoss


def install_checks(directory: pathlib.Path, maximum_splits: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    original_preprocess = LanguageModelBatch._set_target_inputs
    original_loss = LanguageModelLabelEntropyLoss._forward_backward
    checked_batches = 0
    checked_splits = 0
    rank = int(os.environ.get("RANK", "0"))

    def preprocess(batch, inputs, config):
        nonlocal checked_batches
        original_preprocess(batch, inputs, config)
        if batch.is_meta or not config.return_valid_document_count or checked_batches >= maximum_splits:
            return
        tokens = batch.tokens.cpu().tolist()
        spans = (
            batch.loss_masking_spans.ranges
            if config.use_loss_masking_spans and batch.loss_masking_spans is not None
            else []
        )
        for distance in range(1, config.num_labels + 1):
            counts = []
            documents = 0
            total_labels = 0
            offset = 0
            for length in batch.lengths:
                valid = [
                    position
                    for position in range(offset + distance, offset + length)
                    if tokens[position] >= 0 and not any(begin <= position < end for begin, end in spans)
                ]
                count = len(valid)
                counts.extend([count] * length)
                documents += int(count > 0)
                total_labels += count
                offset += length
            assert inputs[0].targets[distance - 1].num_valid_documents == documents
            assert inputs[0].targets[distance - 1].num_labels == total_labels
            for index, model_input in enumerate(inputs):
                target = model_input.targets[distance - 1]
                end = model_input.sequence_k_dim.size + distance
                begin = end - model_input.token_dim.size
                torch.testing.assert_close(target.label_counts.cpu(), torch.tensor(counts[begin:end]))
                if index:
                    assert target.num_valid_documents == 0
                    assert target.num_labels == 0
        checked_batches += 1

    def loss(layer, logits, kwargs, losses=None, split_index=0, grad_logits=None):
        nonlocal checked_splits
        if layer._config.reduction != LanguageModelLabelLossReduction.sample or checked_splits >= maximum_splits:
            return original_loss(layer, logits, kwargs, losses, split_index, grad_logits)
        if layer._vocab_parallel:
            raise NotImplementedError("Training diagnostic currently requires a non-vocabulary-parallel head")
        labels = layer._get_labels(kwargs, split_index)
        counts = layer._prepare_target(kwargs[LanguageModelKwargs.label_counts], split_index)
        divisor = max(kwargs[LanguageModelKwargs.num_valid_documents_in_batch][layer._prediction_distance - 1], 1)
        incoming = layer._get_grad_output(kwargs)
        initial = None if grad_logits is None else grad_logits.clone()
        reference_logits = logits.detach().float().requires_grad_(incoming is not None)
        with torch.enable_grad():
            rows = torch.nn.functional.cross_entropy(
                reference_logits * layer._logits_scale_factor, labels, reduction="none"
            )
            reference_loss = (rows / counts.float().clamp_min(1)).sum() / divisor
            reference_grad = (
                None if incoming is None else torch.autograd.grad(reference_loss * incoming, reference_logits)[0]
            )
        actual_loss, actual_grad = original_loss(layer, logits, kwargs, losses, split_index, grad_logits)
        torch.testing.assert_close(actual_loss.float(), reference_loss.detach(), rtol=2e-4, atol=2e-5)
        maximum_gradient_error = None
        if reference_grad is not None:
            expected_grad = reference_grad.to(logits.dtype)
            if initial is not None:
                expected_grad = initial + expected_grad
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0.02, atol=2e-6)
            maximum_gradient_error = (actual_grad.float() - expected_grad.float()).abs().max().item()
        report = {
            "rank": rank,
            "split_index": split_index,
            "prediction_distance": layer._prediction_distance,
            "labels": labels.cpu(),
            "full_document_label_counts": counts.cpu(),
            "global_document_divisor": divisor,
            "actual_loss": actual_loss.detach().cpu(),
            "reference_loss": reference_loss.detach().cpu(),
            "maximum_gradient_error": maximum_gradient_error,
            "preprocessed_batches_checked": checked_batches,
        }
        torch.save(report, directory / f"rank_{rank}_call_{checked_splits}.pt")
        print(
            f"[sample-loss-check rank={rank} call={checked_splits}] PASS loss={actual_loss.item():.8f} divisor={divisor} max_grad_error={maximum_gradient_error}",
            flush=True,
        )
        checked_splits += 1
        return actual_loss, actual_grad

    LanguageModelBatch._set_target_inputs = preprocess
    LanguageModelLabelEntropyLoss._forward_backward = loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--debug-dir", type=pathlib.Path, default=pathlib.Path("/tmp/fast_llm_tests/training_sample_checks")
    )
    parser.add_argument("--debug-splits", type=int, default=8)
    options, arguments = parser.parse_known_args()
    install_checks(options.debug_dir, options.debug_splits)
    fast_llm_main(arguments)
