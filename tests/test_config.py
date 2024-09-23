import pathlib
import subprocess


def test_validate_without_import():
    repo_path = pathlib.Path(__file__).parents[1].resolve()
    command = [
        "python3",
        "-c",
        "\n".join(
            [
                "import sys",
                "sys.path=[p for p in sys.path if not any(x in p for x in ('site-packages', 'dist-packages', '.egg'))]",
                f"sys.path.append('{repo_path}')",
                "from fast_llm.tools.cli import fast_llm as main",
                "main(['train', 'gpt', '-v', '--data_path=path/to/data'])",
            ]
        ),
    ]

    completed_proc = subprocess.run(command)
    if completed_proc.returncode:
        raise RuntimeError(f"Process failed with return code {completed_proc.returncode}")
