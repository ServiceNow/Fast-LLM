"""Exercise epoch tails, saves, and reproducible resume through the real CLI."""

import os
import shutil
import socket
import subprocess
import sys

import pytest
import torch
import yaml
from safetensors.torch import load_file

from tests.data.test_epoch import write_source


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
@pytest.mark.parametrize("implementation,sdp,reduction", [("sdpa", 2, "token"), ("flash", 1, "sample")])
def test_epoch_training_and_resume(tmp_path, implementation, sdp, reduction):
    if implementation == "flash":
        from fast_llm.layers.attention.attention import _flash_available

        if not _flash_available:
            pytest.skip("Flash attention unavailable")
    # A single packed sequence: in DP=2 mode one rank has only padding slots.
    source = write_source(tmp_path / "data.fast_llm_dataset", [5, 3])
    config = {
        "run": {"experiment_dir": str(tmp_path / "full"), "torch_dynamo_enable": False},
        "model": {
            "distributed": {"compute_dtype": "bfloat16", "sequence_data_parallel": sdp},
            "base_model": {
                "hidden_size": 64,
                "tied_embedding_weight": False,
                "embeddings": {"vocab_size": 128, "vocab_parallel": False},
                "decoder": {
                    "num_blocks": 1,
                    "block": {
                        "normalization": {"type": "rms_norm"},
                        "mixer": {
                            "implementation": implementation,
                            "heads": 4,
                            "head_groups": 2,
                            "head_size": 16,
                            "add_linear_biases": False,
                            "rotary": {"type": "default"},
                        },
                        "mlp": {
                            "intermediate_size": 128,
                            "activation": "silu",
                            "gated": True,
                            "add_linear_biases": False,
                        },
                    },
                },
                "head": {
                    "normalization": {"type": "rms_norm"},
                    "losses": {"cross_entropy": {"type": "label", "reduction": reduction}},
                },
            },
        },
        "data": {
            "micro_batch_size": 16,
            "maximum_document_length": 16,
            "truncate_documents": False,
            "datasets": {"training": {"type": "epoch", "datasets": [source]}},
        },
        "training": {
            "epochs": 2,
            "global_batch_size": 4,
            "num_workers": 0,
            "checkpoint": {"every_epochs": 1, "keep": 2},
            "export": {"every_epochs": 1, "format": "llama"},
        },
        "optimizer": {"learning_rate": {"base": 2e-5, "decay_style": "cosine", "warmup_epochs": 0}},
    }

    def launch(name):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (tmp_path / f"{name}.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node=2",
                    f"--master_port={port}",
                    "-m",
                    "fast_llm.cli",
                    "train",
                    "gpt",
                    "--config",
                    str(path),
                ],
                env={**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        assert result.returncode == 0, (tmp_path / f"{name}.log").read_text()

    launch("full")
    full = tmp_path / "full"
    assert (full / "checkpoint/1/ok").is_file()
    assert (full / "checkpoint/2/ok").is_file()
    resolved = yaml.safe_load((full / "config.yaml").read_text())
    assert resolved["training"]["train_iters"] == 2
    assert resolved["data"]["datasets"]["training"]["plan_summary"]["packed_sequences"] == [1, 1]
    resumed = tmp_path / "resumed"
    (resumed / "checkpoint").mkdir(parents=True)
    shutil.copytree(full / "checkpoint/1", resumed / "checkpoint/1")
    config["run"]["experiment_dir"] = str(resumed)
    launch("resume")
    a = load_file(str(full / "export/llama/2/model_0.safetensors"))
    b = load_file(str(resumed / "export/llama/2/model_0.safetensors"))
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
