import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(model_id: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id}")
    print(f"Saving to {output_dir}")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Model downloaded successfully to {output_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model id",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Local directory where model should be stored",
    )

    args = parser.parse_args()

    download_model(
        model_id=args.model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


"""
python download_model.py \
    --model LiquidAI/LFM2.5-350M \
    --output_dir ./models/LFM2.5-350M
"""