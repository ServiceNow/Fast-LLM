import concurrent.futures
import logging
import os
import pathlib
import re
import shutil
import subprocess

from fast_llm.engine.config_utils.logging import configure_logging

try:
    import hf_transfer  # type: ignore[no-redef]
except ImportError as e:
    raise ImportError("Please install hf_transfer to use this script") from e

try:
    # must be set before importing huggingface_hub and fast_llm
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    import huggingface_hub as hf_hub
    from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER

    assert HF_HUB_ENABLE_HF_TRANSFER, "hf_transfer is not enabled"
    hf_hub.logging.set_verbosity_debug()
except ImportError as e:
    raise ImportError("Please install huggingface_hub to use this script") from e


from fast_llm.config import Config, config_class, Field  # isort:skip
from fast_llm.engine.multi_stage.config import CheckpointType  # isort:skip
from fast_llm.tools.convert import ConversionConfig, convert_model  # isort:skip


logger = logging.getLogger(__name__)


OTHER_TOKENIZER_FILES = ["generation_config.json", "special_tokens_map.json", "tokenizer_config.json"]


@config_class()
class PushConfig(Config):
    experiment_dir: pathlib.Path = Field()
    repo_name: str = Field()
    model_type: str = Field()
    tokenizer_path: pathlib.Path = Field()
    tmp_checkpoint_dir: pathlib.Path | None = Field(default=None)
    use_cpu: bool = Field(default=False)

    def _validate(self):
        super()._validate()
        if self.tmp_checkpoint_dir is None:
            self.tmp_checkpoint_dir: pathlib.Path = self.experiment_dir / "tmp_checkpoint"


def get_iter_number(maybe_iter_number: str) -> None | int:
    m = re.match(r"(\d+)", maybe_iter_number)
    if m is not None:
        return int(m.group(1))
    return None


def get_commited_iter_numbers(hf_repo: hf_hub.Repository) -> list[int]:
    subprocess.run(["git", "fetch", "origin"], cwd=hf_repo.local_dir, capture_output=True)
    commits = subprocess.run(
        ["git", "log", f"origin/main", "--pretty=format:%H %s"], cwd=hf_repo.local_dir, capture_output=True, text=True
    ).stdout.split("\n")
    # Each line is hash and message
    # Just keep the message
    print("commits", commits)
    commits = [c.split(" ", 1)[1] for c in commits]
    # Keep commits corresponding to a new iter
    return [iter_number for c in commits if (iter_number := get_iter_number(c)) is not None]


def git_add_safe_directory(directory: pathlib.Path) -> None:
    cmd_output = subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", str(directory)], capture_output=True
    )
    logger.info(cmd_output)


def copy_tokenizer_files(tokenizer_path: pathlib.Path, tmp_checkpoint_dir: pathlib.Path) -> None:
    logger.info(f"Copying the tokenizer {tokenizer_path} into {tmp_checkpoint_dir}")
    assert tokenizer_path.exists() and tokenizer_path.suffix == ".json"
    shutil.copy(tokenizer_path, tmp_checkpoint_dir / "tokenizer.json")
    for other_file in OTHER_TOKENIZER_FILES:
        fname = tokenizer_path.with_name(other_file)
        if fname.exists():
            logger.info(f"Copying {fname} into {tmp_checkpoint_dir}")
            shutil.copy(fname, tmp_checkpoint_dir / other_file)
        else:
            logger.info(f"File not found: {fname}")


def setup(config: PushConfig) -> hf_hub.HfApi:
    config.to_logs()
    os.environ["HF_TOKEN"] = pathlib.Path(os.environ["HUGGINGFACE_API_KEY_PATH"]).open("r").read().strip()
    hf_api = hf_hub.HfApi()
    git_add_safe_directory(config.tmp_checkpoint_dir)
    hf_api.create_repo(config.repo_name, private=True, exist_ok=True)
    return hf_api


def push_model(config: PushConfig) -> None:
    logging.getLogger().setLevel(logging.INFO)

    # Get Huggingface Hub API object
    hf_api = setup(config)

    # Get list of checkpoints in the repo
    logger.info(f"Cloning repo into {config.tmp_checkpoint_dir}...")
    hf_repo = hf_hub.Repository(local_dir=config.tmp_checkpoint_dir, clone_from=config.repo_name, skip_lfs_files=True)

    # Pull latest changes
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    git_pull_output = subprocess.run(["git", "pull"], cwd=config.tmp_checkpoint_dir, capture_output=True, env=env)
    logger.info(git_pull_output)
    committed_iter_numbers = get_commited_iter_numbers(hf_repo)

    # Get local checkpoints
    export_dir = config.experiment_dir / "export"
    logger.info(f"Looking for checkpoints in {export_dir}...")
    new_checkpoint_paths = [
        (iter_number, p)
        for p in export_dir.glob("*")
        if p.is_dir()
        and (iter_number := get_iter_number(p.name)) is not None
        and p.name == str(iter_number)
        and iter_number not in committed_iter_numbers
    ]
    # Order by iter number
    new_checkpoint_paths = sorted(new_checkpoint_paths, key=lambda x: x[0])

    # Copy tokenizer files
    copy_tokenizer_files(config.tokenizer_path, config.tmp_checkpoint_dir)

    logger.info(f"Pushing {len(new_checkpoint_paths)} checkpoints to {config.repo_name}...")

    # Launch a thread pool to convert the checkpoints in sequence but push and evaluate them in parallel.
    # The thread pool will wait for pushes to be done before shutting down.
    commit_info: None | concurrent.futures.Future[hf_hub.CommitInfo] = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        hf_api._thread_pool = executor

        for _, checkpoint_path in new_checkpoint_paths:
            checkpoint_path_hf = checkpoint_path.with_name(checkpoint_path.name + "_hf")

            # Block until the conversion is done
            convert_model(
                ConversionConfig(
                    input_type=CheckpointType.distributed,
                    output_type=CheckpointType.huggingface,
                    input_path=checkpoint_path,
                    output_path=checkpoint_path_hf,
                    model_type=config.model_type,
                    use_cpu=config.use_cpu,
                    exist_ok=False,  # skip if already processed
                    layers_per_step=(
                        8 if config.model_type == "mixtral" else None
                    ),  # split into 8 layers per step for mixtral
                )
            )

            # Wait for the previous commit to be done before linking the files
            if commit_info is not None:
                _ = commit_info.result()

            # Link all files in the hf checkpoint to the tmp checkpoint (must be in the same filesystem)
            for file in checkpoint_path_hf.iterdir():
                if file.name == "ok":
                    continue
                dest = config.tmp_checkpoint_dir / file.name
                dest.unlink(missing_ok=True)
                dest.hardlink_to(file)
            logger.info(f"Pushing to {config.repo_name}...")

            # Copy metadata
            metadata_path = checkpoint_path / "metadata.yaml"
            if metadata_path.exists():
                metadata_dest = config.tmp_checkpoint_dir / "metadata.yaml"
                shutil.copy(metadata_path, metadata_dest)

            # Commit and push the checkpoint as a future
            commit_info = hf_api.upload_folder(
                folder_path=config.tmp_checkpoint_dir,
                repo_id=config.repo_name,
                commit_message=checkpoint_path.stem,
                run_as_future=True,
            )

    logger.info(f"Removing tmp directory {config.tmp_checkpoint_dir}")
    shutil.rmtree(config.tmp_checkpoint_dir)
    logger.info(f"Done!")


def main(args=None):
    configure_logging()
    config: PushConfig = PushConfig.from_flat_args(args)
    push_model(config)


if __name__ == "__main__":
    main()
