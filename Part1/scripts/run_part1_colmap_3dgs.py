#!/usr/bin/env python3
"""
Unified Part 1 Plan A runner: COLMAP -> 3DGS.

This script merges the six original runners:
  - run_405841.py
  - run_405841_96frames.py
  - run_dl3dv2.py
  - run_dl3dv2_96frames.py
  - run_re10k1.py
  - run_re10k1_96frames.py

Typical usage from CV-Project4/Part1/scripts:
  python run_part1_colmap_3dgs.py --waymo405841
  python run_part1_colmap_3dgs.py --dl3dv2
  python run_part1_colmap_3dgs.py --re10k1

For subsampled datasets:
  python run_part1_colmap_3dgs.py --subsampled --waymo405841 --frame 96
  python run_part1_colmap_3dgs.py --subsampled --dl3dv2 --frame 96
  python run_part1_colmap_3dgs.py --subsampled --re10k1 --frame 96

Default data roots:
  full data:       <project_root>/datasets/
  subsampled data: <project_root>/Part1/subsampled_datasets/

Default output root:
  <project_root>/Part1/output/
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}

WAYMO_INTRINSICS = {
    "fx": 2066.697564417299,
    "fy": 2066.697564417299,
    "cx": 950.5512774150723,
    "cy": 641.1870541472169,
    "k1": 0.036260226499726975,
    "k2": -0.35323300768499216,
    "p1": 0.0003256378403347263,
    "p2": 0.00021755891240936618,
}


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    full_dir_name: str
    subsampled_dir_prefix: str
    output_prefix: str
    full_image_subdir: str
    camera_model: str
    always_upsample_to_512: bool = False
    needs_undistort: bool = False
    camera_params: Optional[str] = None
    needs_images_symlink_from_rgb: bool = False


def waymo_camera_params() -> str:
    p = WAYMO_INTRINSICS
    return f"{p['fx']},{p['fy']},{p['cx']},{p['cy']},{p['k1']},{p['k2']},{p['p1']},{p['p2']}"


DATASETS = {
    "waymo405841": DatasetConfig(
        key="waymo405841",
        full_dir_name="405841",
        subsampled_dir_prefix="405841_colmap",
        output_prefix="405841_colmap",
        full_image_subdir="FRONT/rgb",
        camera_model="OPENCV",
        needs_undistort=True,
        camera_params=waymo_camera_params(),
    ),
    "dl3dv2": DatasetConfig(
        key="dl3dv2",
        full_dir_name="DL3DV-2",
        subsampled_dir_prefix="dl3dv2_colmap",
        output_prefix="dl3dv2_colmap",
        full_image_subdir="rgb",
        camera_model="SIMPLE_PINHOLE",
        needs_images_symlink_from_rgb=True,
    ),
    "re10k1": DatasetConfig(
        key="re10k1",
        full_dir_name="Re10k-1",
        subsampled_dir_prefix="re10k1_colmap",
        output_prefix="re10k1_colmap",
        full_image_subdir="images",
        camera_model="SIMPLE_PINHOLE",
        always_upsample_to_512=True,
    ),
}


def print_block(title: str, command: Optional[Iterable[str]] = None) -> None:
    print(f"\n{'=' * 80}")
    print(f"[STEP] {title}")
    if command is not None:
        print("[CMD]  " + " ".join(str(x) for x in command))
    print("=" * 80, flush=True)


def run_command(command: list[str], desc: str, dry_run: bool = False, cwd: Optional[Path] = None) -> None:
    print_block(desc, command)
    if dry_run:
        return
    result = subprocess.run(command, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(f"[ERROR] Command failed with code {result.returncode}: {' '.join(command)}")


def auto_project_root() -> Path:
    candidates: list[Path] = []
    here = Path.cwd().resolve()
    candidates.extend([here, *here.parents])
    candidates.extend([
        Path.home() / "CV-Project4",
        Path.home() / "CV_PROJECT",
        Path.home() / "CV_Project4",
    ])

    for c in candidates:
        if (c / "datasets").exists() or (c / "Part1" / "subsampled_datasets").exists():
            return c
    return Path.home() / "CV-Project4"


def setup_ld_library_path(python_version: str) -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        return
    torch_lib = Path(conda_prefix) / "lib" / f"python{python_version}" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["LD_LIBRARY_PATH"] = str(torch_lib) + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    files = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS])
    if not files:
        raise RuntimeError(f"No image files found in {image_dir}")
    return files


def ensure_dl3dv_images_link(dataset_dir: Path, image_src: Path) -> None:
    images_path = dataset_dir / "images"
    if images_path.exists():
        return
    try:
        images_path.symlink_to(image_src, target_is_directory=True)
        print(f"Symlink created: {images_path} -> {image_src}", flush=True)
    except OSError as exc:
        print(f"[WARN] Failed to create symlink {images_path} -> {image_src}: {exc}", flush=True)
        print("[WARN] Copying images instead so 3DGS can find dataset/images.", flush=True)
        shutil.copytree(image_src, images_path)


def prepare_colmap_images(
    image_src: Path,
    image_colmap: Path,
    always_upsample_to_512: bool,
    dry_run: bool = False,
) -> Path:
    files = list_images(image_src)
    sample = Image.open(files[0]).size
    print(f"  Found {len(files)} images in {image_src}", flush=True)
    print(f"  Sample size: {sample[0]} x {sample[1]}", flush=True)

    should_upsample = always_upsample_to_512 or sample[0] < 512 or sample[1] < 512
    if not should_upsample:
        print("  Images are large enough; COLMAP will use the original image directory.", flush=True)
        return image_src

    image_colmap.mkdir(parents=True, exist_ok=True)
    print(f"  Upsampling/copying COLMAP images to {image_colmap}", flush=True)
    if dry_run:
        return image_colmap

    for src in files:
        dst = image_colmap / src.name
        if dst.exists():
            continue
        img = Image.open(src).convert("RGB").resize((512, 512), Image.LANCZOS)
        img.save(dst)

    copied = list_images(image_colmap)
    print(f"  Prepared {len(copied)} images for COLMAP.", flush=True)
    return image_colmap


def find_best_model(sparse_ws: Path) -> str:
    best_model: Optional[str] = None
    best_count = -1

    for model_dir in sorted([p for p in sparse_ws.iterdir() if p.is_dir()]):
        result = subprocess.run(
            ["colmap", "model_analyzer", "--path", str(model_dir)],
            capture_output=True,
            text=True,
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        match = re.search(r"Registered images:\s*(\d+)", output)
        if not match:
            continue
        count = int(match.group(1))
        print(f"  sparse/{model_dir.name}: {count} registered images", flush=True)
        if count > best_count:
            best_count = count
            best_model = model_dir.name

    if best_model is None:
        raise RuntimeError(f"No valid COLMAP sparse model found in {sparse_ws}")

    print(f"  Best model: sparse/{best_model} ({best_count} registered images)", flush=True)
    return best_model


def copy_sparse_model(src_model: Path, sparse_out: Path) -> None:
    sparse_out.mkdir(parents=True, exist_ok=True)
    for filename in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = src_model / filename
        dst = sparse_out / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing COLMAP model file: {src}")
        shutil.copy2(src, dst)
        print(f"  Copied {filename} -> {dst}", flush=True)


def fix_undistorted_sparse_structure(undistorted: Path) -> None:
    undist_sparse = undistorted / "sparse"
    undist_sparse_0 = undist_sparse / "0"
    undist_sparse_0.mkdir(parents=True, exist_ok=True)

    for filename in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = undist_sparse / filename
        dst = undist_sparse_0 / filename
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  Moved {filename} -> sparse/0/", flush=True)
        elif dst.exists():
            print(f"  {filename} already exists in sparse/0/", flush=True)
        else:
            print(f"  [WARN] {filename} was not found in {undist_sparse}", flush=True)


def resolve_paths(args: argparse.Namespace, cfg: DatasetConfig) -> dict[str, Path]:
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else auto_project_root()
    gs_repo = Path(args.gs_repo).expanduser().resolve() if args.gs_repo else project_root / "gaussian-splatting"
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else project_root / "Part1" / "output"

    if args.subsampled:
        dataset_name = f"{cfg.subsampled_dir_prefix}_{args.frame}frames"
        dataset_dir = project_root / "Part1" / "subsampled_datasets" / dataset_name
        image_src = dataset_dir / "images"
        output_dir = output_root / dataset_name
    else:
        dataset_dir = project_root / "datasets" / cfg.full_dir_name
        image_src = dataset_dir / cfg.full_image_subdir
        output_dir = output_root / cfg.output_prefix

    return {
        "project_root": project_root,
        "gs_repo": gs_repo,
        "dataset_dir": dataset_dir,
        "image_src": image_src,
        "image_colmap": dataset_dir / "images_colmap",
        "colmap_ws": dataset_dir / "colmap_ws",
        "sparse_ws": dataset_dir / "colmap_ws" / "sparse",
        "sparse_out": dataset_dir / "sparse" / "0",
        "undistorted": dataset_dir / "undistorted",
        "output_dir": output_dir,
        "database": dataset_dir / "colmap_ws" / "database.db",
        "logs": project_root / "Part1" / "logs",
    }


def run_pipeline(args: argparse.Namespace, cfg: DatasetConfig) -> None:
    setup_ld_library_path(args.python_version)
    paths = resolve_paths(args, cfg)

    print("\n[CONFIG]", flush=True)
    for key in ["project_root", "gs_repo", "dataset_dir", "image_src", "output_dir"]:
        print(f"  {key}: {paths[key]}", flush=True)
    print(f"  dataset: {cfg.key}", flush=True)
    print(f"  subsampled: {args.subsampled}", flush=True)

    if not paths["gs_repo"].exists():
        raise FileNotFoundError(f"gaussian-splatting repository not found: {paths['gs_repo']}")

    if cfg.needs_images_symlink_from_rgb and not args.subsampled:
        ensure_dl3dv_images_link(paths["dataset_dir"], paths["image_src"])

    for key in ["image_colmap", "sparse_ws", "sparse_out", "output_dir", "logs"]:
        if not args.dry_run:
            paths[key].mkdir(parents=True, exist_ok=True)

    if args.clean_colmap and not args.dry_run:
        for key in ["colmap_ws", "sparse_out", "image_colmap", "undistorted"]:
            target = paths[key]
            if target.exists():
                print(f"[CLEAN] Removing {target}", flush=True)
                shutil.rmtree(target)
        for key in ["image_colmap", "sparse_ws", "sparse_out"]:
            paths[key].mkdir(parents=True, exist_ok=True)

    print_block("Preparing images for COLMAP")
    colmap_img_dir = prepare_colmap_images(
        paths["image_src"],
        paths["image_colmap"],
        always_upsample_to_512=cfg.always_upsample_to_512,
        dry_run=args.dry_run,
    )

    feature_cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(paths["database"]),
        "--image_path", str(colmap_img_dir),
        "--ImageReader.camera_model", cfg.camera_model,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", str(args.max_num_features),
    ]
    if cfg.camera_params:
        feature_cmd.extend(["--ImageReader.camera_params", cfg.camera_params])

    run_command(feature_cmd, f"COLMAP feature extraction ({cfg.camera_model})", args.dry_run)

    run_command(
        [
            "colmap", "sequential_matcher",
            "--database_path", str(paths["database"]),
            "--SequentialMatching.overlap", str(args.overlap),
            "--SequentialMatching.quadratic_overlap", "1" if args.quadratic_overlap else "0",
        ],
        "COLMAP sequential matching",
        args.dry_run,
    )

    run_command(
        [
            "colmap", "mapper",
            "--database_path", str(paths["database"]),
            "--image_path", str(colmap_img_dir),
            "--output_path", str(paths["sparse_ws"]),
        ],
        "COLMAP mapper (SfM)",
        args.dry_run,
    )

    if args.dry_run:
        print("\n[DRY RUN] Stopping before model analysis/training.", flush=True)
        return

    print_block("Selecting best COLMAP sparse sub-model")
    best_model = find_best_model(paths["sparse_ws"])

    print_block("Copying selected sparse model to sparse/0")
    copy_sparse_model(paths["sparse_ws"] / best_model, paths["sparse_out"])

    train_source = paths["dataset_dir"]
    if cfg.needs_undistort:
        run_command(
            [
                "colmap", "image_undistorter",
                "--image_path", str(colmap_img_dir),
                "--input_path", str(paths["sparse_out"]),
                "--output_path", str(paths["undistorted"]),
                "--output_type", "COLMAP",
            ],
            "COLMAP image undistortion (OPENCV -> PINHOLE)",
            args.dry_run,
        )
        print_block("Fixing undistorted sparse/0 structure")
        fix_undistorted_sparse_structure(paths["undistorted"])
        train_source = paths["undistorted"]

    run_command(
        [
            sys.executable,
            "train.py",
            "-s", str(train_source),
            "-m", str(paths["output_dir"]),
            "--eval",
            "--iterations", str(args.iterations),
        ],
        "3DGS training",
        args.dry_run,
        cwd=paths["gs_repo"],
    )

    if not args.skip_render:
        run_command(
            [sys.executable, "render.py", "-m", str(paths["output_dir"])],
            "Rendering",
            args.dry_run,
            cwd=paths["gs_repo"],
        )

    if not args.skip_metrics:
        run_command(
            [sys.executable, "metrics.py", "-m", str(paths["output_dir"])],
            "Metrics evaluation",
            args.dry_run,
            cwd=paths["gs_repo"],
        )

    print(f"\n[DONE] {cfg.key} complete. Output: {paths['output_dir']}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified COLMAP -> 3DGS runner for Project 4 Part 1 datasets."
    )

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--waymo405841", action="store_true", help="Run Waymo-405841 pipeline.")
    data_group.add_argument("--dl3dv2", action="store_true", help="Run DL3DV-2 pipeline.")
    data_group.add_argument("--re10k1", action="store_true", help="Run Re10k-1 pipeline.")

    parser.add_argument("--subsampled", action="store_true", help="Read from Part1/subsampled_datasets instead of datasets.")
    parser.add_argument("--frame", type=int, default=96, help="Subsampled frame count suffix, e.g. 96 for *_96frames.")
    parser.add_argument("--project-root", type=str, default=None, help="Path to CV-Project4. Default: auto-detect or ~/CV-Project4.")
    parser.add_argument("--gs-repo", type=str, default=None, help="Path to gaussian-splatting repo. Default: <project-root>/gaussian-splatting.")
    parser.add_argument("--output-root", type=str, default=None, help="Output root. Default: <project-root>/Part1/output.")

    parser.add_argument("--iterations", type=int, default=30000, help="3DGS training iterations.")
    parser.add_argument("--max-num-features", type=int, default=4096, help="COLMAP SIFT max features.")
    parser.add_argument("--overlap", type=int, default=10, help="COLMAP sequential matcher overlap.")
    parser.add_argument("--no-quadratic-overlap", dest="quadratic_overlap", action="store_false", help="Disable COLMAP quadratic overlap.")
    parser.set_defaults(quadratic_overlap=True)

    parser.add_argument("--python-version", type=str, default="3.10", help="Python version used in CONDA_PREFIX torch lib path.")
    parser.add_argument("--clean-colmap", action="store_true", help="Remove existing COLMAP workspace/images_colmap/sparse/undistorted before running.")
    parser.add_argument("--skip-render", action="store_true", help="Skip render.py after training.")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip metrics.py after training.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and validate paths without executing COLMAP/3DGS.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.waymo405841:
        cfg = DATASETS["waymo405841"]
    elif args.dl3dv2:
        cfg = DATASETS["dl3dv2"]
    elif args.re10k1:
        cfg = DATASETS["re10k1"]
    else:
        raise SystemExit("Please choose one dataset flag.")

    run_pipeline(args, cfg)


if __name__ == "__main__":
    main()
