# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Download VoiceBank-DEMAND-16k dataset for speech enhancement evaluation.

Source: JacobLinCool/VoiceBank-DEMAND-16k on Hugging Face Hub
Output: data/datasets/VoiceBank-DEMAND-16k/{train,test}/{clean,noisy}/

Dependencies (not in main package — install separately):
    pip install datasets

Usage:
    python3 examples/audio/scripts/download_voicebank_demand.py
    python3 examples/audio/scripts/download_voicebank_demand.py --dataset_path /path/to/output
    python3 examples/audio/scripts/download_voicebank_demand.py --force_download
"""

import os
import argparse

try:
    from datasets import load_dataset, Audio
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required package 'datasets' is not installed.")
    print("Run in the virtual environment: pip install datasets")
    print("Then re-run: python3 examples/audio/scripts/download_voicebank_demand.py")
    raise SystemExit(1)


HF_DATASET_NAME = "JacobLinCool/VoiceBank-DEMAND-16k"


def download_voicebank(dataset_path: str, force_download: bool = False) -> None:
    # Resolve relative paths from repo root (script lives at examples/audio/scripts/)
    if not os.path.isabs(dataset_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, "../../.."))
        dataset_path = os.path.join(repo_root, dataset_path)

    # Skip if already downloaded
    test_noisy = os.path.join(dataset_path, "test", "noisy")
    if os.path.isdir(test_noisy) and os.listdir(test_noisy) and not force_download:
        print(f"INFO: VoiceBank-DEMAND-16k already exists at {dataset_path} — skipping.")
        print(f"      Use --force_download to re-download.")
        return

    os.makedirs(dataset_path, exist_ok=True)

    print(f"INFO: Downloading {HF_DATASET_NAME} from Hugging Face...")
    print(f"      Destination: {dataset_path}")
    print(f"      Expected size: ~2 GB\n")

    dataset = load_dataset(HF_DATASET_NAME)

    # Use decode=False to avoid torchcodec/FFmpeg dependencies — get raw bytes directly
    dataset = dataset.cast_column("clean", Audio(decode=False))
    dataset = dataset.cast_column("noisy", Audio(decode=False))

    print(f"INFO: Splits: {list(dataset.keys())}")
    for split_name in dataset:
        print(f"      {split_name}: {len(dataset[split_name])} samples")
    print()

    for split_name, split_data in dataset.items():
        clean_dir = os.path.join(dataset_path, split_name, "clean")
        noisy_dir = os.path.join(dataset_path, split_name, "noisy")
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(noisy_dir, exist_ok=True)

        print(f"INFO: Saving {split_name} split...")
        for item in tqdm(split_data, desc=f"  {split_name}", unit="files"):
            filename = f"{item['id']}.wav"
            with open(os.path.join(clean_dir, filename), "wb") as f:
                f.write(item["clean"]["bytes"])
            with open(os.path.join(noisy_dir, filename), "wb") as f:
                f.write(item["noisy"]["bytes"])

    print(f"\nINFO: VoiceBank-DEMAND-16k saved to {dataset_path}")
    print(f"      Structure: {{train,test}}/{{clean,noisy}}/")


def main():
    # Default path is relative to repo root
    default_path = os.path.join("data", "datasets", "VoiceBank-DEMAND-16k")

    parser = argparse.ArgumentParser(
        description="Download VoiceBank-DEMAND-16k dataset for speech enhancement"
    )
    parser.add_argument(
        "--dataset_path",
        default=default_path,
        help=f"Output directory (default: {default_path}, relative to repo root)",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download even if the dataset already exists",
    )
    args = parser.parse_args()

    download_voicebank(args.dataset_path, force_download=args.force_download)


if __name__ == "__main__":
    main()
