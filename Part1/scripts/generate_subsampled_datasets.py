#!/usr/bin/env python3
"""
Generate uniformly subsampled datasets for Project 4 Part 1.

Expected project layout:

CV-Project4/
├── datasets/
│   ├── 405841/images/
│   ├── DL3DV-2/images/
│   └── Re10k-1/images/
└── Part1/
    ├── scripts/
    │   └── generate_subsampled_datasets.py
    └── subsampled_datasets/

Example usage from CV-Project4/Part1/scripts:

    python generate_subsampled_datasets.py --frame 96
    python generate_subsampled_datasets.py --frame 72 --mode symlink
    python generate_subsampled_datasets.py --frame 96 --datasets 405841 DL3DV-2

The script creates one subsampled dataset per reconstruction method:

    CV-Project4/Part1/subsampled_datasets/{dataset}_{method}_{N}frames/images

By default, the same selected frame set is reused for COLMAP and VGGT to keep
initialization comparisons fair.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

VALID_EXTS = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}

DATASET_NAME_MAP = {
    "405841": "405841",
    "DL3DV-2": "dl3dv2",
    "Re10k-1": "re10k1",
}

DEFAULT_METHODS = ("colmap", "vggt")


def infer_project_root() -> Path:
    """Infer CV-Project4 root from a script placed in CV-Project4/Part1/scripts."""
    return Path(__file__).resolve().parents[2]


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Expected an image directory, got: {images_dir}")

    return sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix in VALID_EXTS],
        key=lambda p: p.name,
    )


def select_by_target_count(image_paths: Sequence[Path], target_count: int) -> List[Path]:
    """Uniformly sample target_count frames while preserving first and last frames."""
    total = len(image_paths)
    if total == 0:
        return []
    if target_count <= 0:
        raise ValueError(f"--frame must be positive, got {target_count}")
    if total <= target_count:
        return list(image_paths)
    if target_count == 1:
        return [image_paths[0]]

    indices = [round(i * (total - 1) / (target_count - 1)) for i in range(target_count)]
    indices = sorted(set(indices))
    return [image_paths[i] for i in indices]


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_images(selected_paths: Iterable[Path], dst_images_dir: Path, mode: str, overwrite: bool) -> None:
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        clear_directory(dst_images_dir)

    for src in selected_paths:
        dst = dst_images_dir / src.name
        if dst.exists() or dst.is_symlink():
            if overwrite:
                dst.unlink()
            else:
                raise FileExistsError(
                    f"Destination file already exists: {dst}. Use --overwrite to replace it."
                )

        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "symlink":
            os.symlink(src.resolve(), dst)
        else:
            raise ValueError(f"Unsupported mode: {mode}")


def preview_selection(dataset_name: str, image_paths: Sequence[Path], selected: Sequence[Path], target_count: int) -> None:
    print(f"[{dataset_name}] total={len(image_paths)}, selected={len(selected)}, target={target_count}")
    if selected:
        first = [p.name for p in selected[:5]]
        last = [p.name for p in selected[-5:]]
        print(f"  first: {first}")
        print(f"  last : {last}")
    print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subsample CV-Project4 datasets for Part 1 COLMAP/VGGT experiments."
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=96,
        help="Target number of frames per dataset. If a dataset has fewer frames, all frames are kept. Default: 96.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Path to CV-Project4. Default: inferred from this script location.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_NAME_MAP.keys()),
        choices=list(DATASET_NAME_MAP.keys()),
        help="Datasets to subsample. Default: all mandatory datasets.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Output method suffixes to generate. Default: colmap vggt.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Whether to copy images or create symbolic links. Default: copy.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Default: CV-Project4/Part1/subsampled_datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only preview selected frames and output paths without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files inside output images directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve() if args.project_root else infer_project_root()
    datasets_root = project_root / "datasets"
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else project_root / "Part1" / "subsampled_datasets"
    )

    print(f"PROJECT_ROOT  = {project_root}")
    print(f"DATASETS_ROOT = {datasets_root}")
    print(f"OUTPUT_ROOT   = {output_root}")
    print(f"TARGET_FRAMES = {args.frame}")
    print(f"WRITE_MODE    = {args.mode}")
    print()

    created_dirs: List[Path] = []

    for dataset_folder_name in args.datasets:
        output_dataset_name = DATASET_NAME_MAP[dataset_folder_name]
        images_dir = datasets_root / dataset_folder_name / "images"
        image_paths = list_images(images_dir)
        selected = select_by_target_count(image_paths, args.frame)
        selected_count = len(selected)

        preview_selection(dataset_folder_name, image_paths, selected, args.frame)

        for method in args.methods:
            out_dir = output_root / f"{output_dataset_name}_{method}_{selected_count}frames"
            out_images_dir = out_dir / "images"

            if args.dry_run:
                print(f"[dry-run] would create: {out_images_dir}")
            else:
                write_images(selected, out_images_dir, mode=args.mode, overwrite=args.overwrite)
                created_dirs.append(out_dir)
                print(f"Created: {out_dir}")
        print()

    if args.dry_run:
        print("Dry run finished. No files were written.")
    else:
        print("Done. Output summary:")
        for out_dir in created_dirs:
            imgs = list_images(out_dir / "images")
            print(f"  {out_dir.name}: {len(imgs)} images")


if __name__ == "__main__":
    main()
