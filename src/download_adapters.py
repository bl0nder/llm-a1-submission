import argparse
import os
import shutil

from huggingface_hub import snapshot_download

from peft_sweep import get_logger

log = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="adapters")
    return parser.parse_args()


def main():
    args = parse_args()
    log.info(f"Downloading all adapters from {args.repo_id} into {args.adapter_path}")
    # The repo stores every adapter under adapters/, so copy that folder's contents
    # into --adapter_path to get adapter_path/A1, adapter_path/A2, ...
    snapshot_dir = snapshot_download(args.repo_id, repo_type="model", allow_patterns=["adapters/*"])
    shutil.copytree(os.path.join(snapshot_dir, "adapters"), args.adapter_path, dirs_exist_ok=True)
    log.info("Done")


if __name__ == "__main__":
    main()
