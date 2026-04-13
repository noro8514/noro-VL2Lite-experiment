#!/usr/bin/env python3
"""Prepare Stanford Dogs dataset for this repo.

This script downloads the Stanford Dogs images archive (optional), extracts it
into the target directory, and creates `train_list.mat` and `test_list.mat`
containing `file_list` and `labels` variables in the MATLAB-compatible format
expected by `src.data.components.kd_dataloader.StanfordDogsDataset`.

Usage:
  python scripts/prepare_stanford_dogs.py --out-dir PATH [--download]

If `--download` is provided the script will try to download the official
`images.tar` from the Stanford website. If the archive is already present or
extracted the script will skip downloading and extraction and only create the
.mat split files.
"""
from __future__ import annotations

import argparse
import os
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import savemat


DEFAULT_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Archive already exists at {dest}, skipping download.")
        return
    print(f"Downloading {url} -> {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print("Download completed.")


def extract_tar(tar_path: Path, extract_to: Path) -> None:
    # Only skip extraction if the expected Images/ directory already exists.
    images_dir = extract_to / "Images"
    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"Directory {images_dir} already exists and is not empty, skipping extraction.")
        return
    print(f"Extracting {tar_path} -> {extract_to} ...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_to)
    print("Extraction completed.")


def build_splits(images_dir: Path, train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Walk `images_dir`, split per-class and return lists for train and test.

    Returns file paths relative to `images_dir` (posix-style) and 1-based labels,
    matching the expectations of the original dataset .mat files.
    """
    random.seed(seed)
    class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories found in {images_dir}")

    train_files = []
    train_labels = []
    test_files = []
    test_labels = []

    for class_idx, class_dir in enumerate(class_dirs, start=1):
        imgs = sorted([p.name for p in class_dir.iterdir() if p.is_file()])
        if not imgs:
            continue
        random.shuffle(imgs)
        split_idx = int(len(imgs) * train_ratio)
        train_imgs = imgs[:split_idx]
        test_imgs = imgs[split_idx:]

        rel_prefix = class_dir.name
        train_files.extend([f"{rel_prefix}/{n}" for n in train_imgs])
        train_labels.extend([class_idx] * len(train_imgs))

        test_files.extend([f"{rel_prefix}/{n}" for n in test_imgs])
        test_labels.extend([class_idx] * len(test_imgs))

    return train_files, train_labels, test_files, test_labels


def save_mat_lists(file_list: List[str], labels: List[int], out_path: Path) -> None:
    # MATLAB cell array compatible shape: (N,1) object array
    arr = np.empty((len(file_list), 1), dtype=object)
    for i, s in enumerate(file_list):
        arr[i, 0] = s

    labs = np.array([[int(l)] for l in labels], dtype=np.int32)

    savemat(out_path, {"file_list": arr, "labels": labs})
    print(f"Saved {out_path} (N={len(file_list)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for Stanford Dogs dataset (Images + .mat files).")
    parser.add_argument("--download", action="store_true", help="Download the official images.tar archive.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="URL to download images archive from.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Per-class train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .mat files if present.")
    args = parser.parse_args()

    # Default out dir: PROJECT_ROOT/data/kd_datasets/5_StanfordDogs
    if args.out_dir is None:
        repo_root = Path(__file__).resolve().parents[1]
        args.out_dir = repo_root / "data" / "kd_datasets" / "5_StanfordDogs"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.out_dir / "images.tar"
    images_dir = args.out_dir / "Images"

    if args.download:
        download_file(args.url, archive_path)
        extract_tar(archive_path, args.out_dir)
    else:
        if not images_dir.exists():
            # user didn't request download and Images is missing
            raise RuntimeError(f"Images directory not found at {images_dir}. Run with --download or place Images/ there manually.")

    # After extraction, some archives may unpack into a nested folder. Attempt to
    # discover the actual `Images/` directory if present anywhere under out_dir.
    if not images_dir.exists():
        print(f"Expected {images_dir} not found; searching for an 'Images' directory under {args.out_dir}...")
        found = None
        for p in args.out_dir.rglob('Images'):
            if p.is_dir():
                found = p
                break
        if found:
            print(f"Found Images directory at {found}; using that path.")
            images_dir = found
        else:
            raise RuntimeError(
                f"Images directory not found under {args.out_dir}. "
                "If you used --download, ensure the archive downloaded and extracted correctly. "
                "Otherwise, place the dataset's 'Images/' folder in the output directory or run with --download to fetch it."
            )

    # Build splits
    print(f"Building train/test splits from {images_dir} ...")
    train_files, train_labels, test_files, test_labels = build_splits(images_dir, args.train_ratio, args.seed)

    train_mat = args.out_dir / "train_list.mat"
    test_mat = args.out_dir / "test_list.mat"

    if train_mat.exists() and not args.force:
        print(f"{train_mat} exists. Use --force to overwrite.")
    else:
        save_mat_lists(train_files, train_labels, train_mat)

    if test_mat.exists() and not args.force:
        print(f"{test_mat} exists. Use --force to overwrite.")
    else:
        save_mat_lists(test_files, test_labels, test_mat)

    print("Done. Stanford Dogs prepared.")


if __name__ == "__main__":
    main()
