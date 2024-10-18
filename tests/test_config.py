import pathlib
import subprocess
import yaml
from fast_llm.models.auto import trainer_registry


def test_validate_without_import():
    # Make sure validation imports only the bare minimum.
    # Run the test in a separate process since lots of things are already imported in this one.
    repo_path = pathlib.Path(__file__).parents[1].resolve()
    command = [
        "python3",
        "-c",
        "\n".join(
            [
                # Import required third party libraries here, so they can be found later.
                "import sys, yaml, requests",
                # Prevent any other third party package from being imported (or at least try to)
                "sys.path=[p for p in sys.path if not any(x in p for x in ('site-packages', 'dist-packages', '.egg'))]",
                # We still want to enable imports from within Fast-llm
                f"sys.path.append('{repo_path}')",
                "from fast_llm.tools.cli import fast_llm as main",
                "main(['train', 'gpt', '-v'])",
            ]
        ),
    ]

    completed_proc = subprocess.run(command)
    if completed_proc.returncode:
        raise RuntimeError(f"Process failed with return code {completed_proc.returncode}")


def test_validate_example_config():
    fast_llm_config_dict = yaml.safe_load(
        (pathlib.Path(__file__).parents[1] / "examples" / "mistral-4-node-benchmark.yaml").read_text()
    )
    trainer_registry["gpt"].from_dict(fast_llm_config_dict)
