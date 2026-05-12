#!/usr/bin/env python3
"""
Generate Part 1 aggregate result visualization figures.

This script is converted from part1_results_visualization.ipynb and is meant to
be placed under:
    CV-Project4/Part1/scripts/

Default paths, when run from scripts/:
    results input : CV-Project4/Part1/output/
    figures output: CV-Project4/Part1/figures/

Usage:
    python part1_results_visualization.py
    python part1_results_visualization.py --show
    python part1_results_visualization.py --project-root ~/CV-Project4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

DATASETS = ["405841", "dl3dv2", "re10k1"]
METRICS = ["SSIM", "PSNR", "LPIPS"]
COMPARE_COLORS = ["#4C78A8", "#83B5D5"]
TEXT_COLOR = "#333333"
GRID_ALPHA = 0.22

FOLDERS = {
    # Full-frame, COLMAP + 3DGS
    ("405841", "colmap", "full", "3dgs"): "405841_colmap",
    ("dl3dv2", "colmap", "full", "3dgs"): "dl3dv2_colmap",
    ("re10k1", "colmap", "full", "3dgs"): "re10k1_colmap",
    # 96 frames, 3DGS
    ("405841", "colmap", "96frames", "3dgs"): "405841_colmap_96frames",
    ("405841", "vggt", "96frames", "3dgs"): "405841_vggt_96frames",
    ("dl3dv2", "colmap", "96frames", "3dgs"): "dl3dv2_colmap_96frames",
    ("dl3dv2", "vggt", "96frames", "3dgs"): "dl3dv2_vggt_96frames",
    ("re10k1", "colmap", "96frames", "3dgs"): "re10k1_colmap_96frames",
    ("re10k1", "vggt", "96frames", "3dgs"): "re10k1_vggt_96frames",
    # 96 frames, Wavelet-GS
    ("405841", "colmap", "96frames", "wavelet"): "405841_colmap_96frames_wavelet",
    ("405841", "vggt", "96frames", "wavelet"): "405841_vggt_96frames_wavelet",
    ("dl3dv2", "colmap", "96frames", "wavelet"): "dl3dv2_colmap_96frames_wavelet",
    ("dl3dv2", "vggt", "96frames", "wavelet"): "dl3dv2_vggt_96frames_wavelet",
    ("re10k1", "colmap", "96frames", "wavelet"): "re10k1_colmap_96frames_wavelet",
    ("re10k1", "vggt", "96frames", "wavelet"): "re10k1_vggt_96frames_wavelet",
}

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 10


def find_project_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if parent.name in {"CV-Project4", "CV_PROJECT"}:
            return parent
        if (parent / "Part1").exists() and (parent / "datasets").exists():
            return parent
    for candidate in [Path.cwd(), Path.home() / "CV-Project4", Path.home() / "CV_PROJECT"]:
        if (candidate / "Part1").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Part 1 aggregate result figures.")
    parser.add_argument("--project-root", type=str, default=None, help="Path to CV-Project4. Auto-detected by default.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory containing result folders. Default: <project-root>/Part1/output.")
    parser.add_argument("--figures-dir", type=str, default=None, help="Directory to save figures. Default: <project-root>/Part1/figures.")
    parser.add_argument("--show", action="store_true", help="Also open figures interactively if supported.")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png", help="Output figure format.")
    return parser.parse_args()


def get_folder(dataset: str, init_method: str, frame_setting: str, optimizer: str) -> Optional[str]:
    return FOLDERS.get((dataset, init_method, frame_setting, optimizer))


def _extract_step_from_key(key: str) -> Optional[int]:
    match = re.fullmatch(r"ours_(\d+)", str(key))
    return int(match.group(1)) if match else None


def _select_results_block(raw: object) -> Tuple[Optional[Tuple[str, dict]], Optional[str]]:
    if not isinstance(raw, dict):
        return None, "results.json is not a dict"

    candidates = []
    for key, value in raw.items():
        step = _extract_step_from_key(str(key))
        if step is None or not isinstance(value, dict):
            continue
        if all(metric in value for metric in METRICS):
            candidates.append((step, str(key), value))

    if not candidates:
        return None, f"no valid ours_* block with complete metrics found; keys={list(raw.keys())}"

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, selected_key, selected_block = candidates[0]
    return (selected_key, selected_block), None


def load_results_json(base_dir: Path, folder_name: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
    if folder_name is None:
        return None, "folder mapping missing"

    folder_path = base_dir / folder_name
    json_path = folder_path / "results.json"

    if not folder_path.exists():
        return None, f"folder not found: {folder_path}"
    if not json_path.exists():
        return None, f"results.json not found: {json_path}"

    try:
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        selected, err = _select_results_block(raw)
        if selected is None:
            return None, err
        selected_key, block = selected
        return {"metrics": {m: float(block[m]) for m in METRICS}, "selected_key": selected_key}, None
    except Exception as exc:
        return None, f"failed to read {json_path}: {exc}"


def print_availability_table(base_dir: Path):
    rows = []
    for key, folder in FOLDERS.items():
        dataset, init_method, frame_setting, optimizer = key
        result, err = load_results_json(base_dir, folder)
        rows.append({
            "dataset": dataset,
            "init": init_method,
            "frames": frame_setting,
            "optimizer": optimizer,
            "folder": folder,
            "available": result is not None,
            "selected_key": result["selected_key"] if result is not None else None,
            "note": "ok" if result is not None else err,
        })
    return rows


def _annotate_value(ax, bar, value: float, fontsize: int = 8) -> None:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=fontsize, color=TEXT_COLOR, clip_on=False)


def _draw_no_result(ax, x: float, width: float, label_y: float) -> None:
    ax.bar(x, 0.0, width=width, color="#EAEAEA", edgecolor="#BDBDBD", linewidth=0.8, hatch="//")
    ax.text(x, label_y, "no results", ha="center", va="bottom", rotation=90, fontsize=8, color="#666666")


def _safe_metric(res: Optional[dict], metric: str) -> Optional[float]:
    return None if res is None else float(res["metrics"][metric])


def make_figure(base_dir: Path, compare_mode: str, compare_value: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), dpi=300, constrained_layout=True)

    if compare_mode == "init":
        optimizer = compare_value
        title = f"96-Frame Initialization Comparison ({'3DGS' if optimizer == '3dgs' else 'Wavelet-GS'})"
        specs = [
            {"label": "COLMAP", "init": "colmap", "frames": "96frames", "optimizer": optimizer, "color": COMPARE_COLORS[0]},
            {"label": "VGGT", "init": "vggt", "frames": "96frames", "optimizer": optimizer, "color": COMPARE_COLORS[1]},
        ]
    elif compare_mode == "optimizer":
        init_method = compare_value
        title = f"96-Frame Optimization Comparison ({'COLMAP' if init_method == 'colmap' else 'VGGT'} Init)"
        specs = [
            {"label": "3DGS", "init": init_method, "frames": "96frames", "optimizer": "3dgs", "color": COMPARE_COLORS[0]},
            {"label": "Wavelet-GS", "init": init_method, "frames": "96frames", "optimizer": "wavelet", "color": COMPARE_COLORS[1]},
        ]
    elif compare_mode == "frames":
        title = "COLMAP + 3DGS: Full Frames vs 96 Frames"
        specs = [
            {"label": "Full Frames", "init": "colmap", "frames": "full", "optimizer": "3dgs", "color": COMPARE_COLORS[0]},
            {"label": "96 Frames", "init": "colmap", "frames": "96frames", "optimizer": "3dgs", "color": COMPARE_COLORS[1]},
        ]
    else:
        raise ValueError(f"Unsupported compare_mode: {compare_mode}")

    dataset_records = []
    for dataset in DATASETS:
        local_specs = []
        for spec in specs:
            folder = get_folder(dataset, str(spec["init"]), str(spec["frames"]), str(spec["optimizer"]))
            result, err = load_results_json(base_dir, folder)
            local_specs.append({**spec, "folder": folder, "results": result, "error": err})
        dataset_records.append({"dataset": dataset, "specs": local_specs})

    width = 0.32
    x_base = np.arange(len(DATASETS))

    for ax, metric in zip(axes, METRICS):
        metric_vals = []
        for record in dataset_records:
            for spec in record["specs"]:
                val = _safe_metric(spec["results"], metric)
                if val is not None:
                    metric_vals.append(val)

        ymax = max(metric_vals) * 1.18 if metric_vals else 1.0
        label_y = ymax * 0.03

        for i, base_spec in enumerate(specs):
            offsets = x_base + (i - (len(specs) - 1) / 2) * width
            for j, record in enumerate(dataset_records):
                spec = record["specs"][i]
                val = _safe_metric(spec["results"], metric)
                x = offsets[j]
                if val is None:
                    _draw_no_result(ax, x, width, label_y)
                else:
                    bar = ax.bar(x, val, width=width, color=base_spec["color"], edgecolor="white", linewidth=0.9, label=base_spec["label"] if j == 0 else None)[0]
                    _annotate_value(ax, bar, val, fontsize=8)

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(x_base)
        ax.set_xticklabels(DATASETS, fontsize=10)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
        ax.set_axisbelow(True)
        legend_handles = [
            Patch(facecolor=specs[0]["color"], edgecolor="white", label=specs[0]["label"]),
            Patch(facecolor=specs[1]["color"], edgecolor="white", label=specs[1]["label"]),
        ]
        ax.legend(handles=legend_handles, fontsize=9, frameon=False, loc="upper left")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    return fig, axes


def main() -> None:
    args = parse_args()
    project_root = find_project_root(args.project_root)
    base_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else project_root / "Part1" / "output"
    figures_dir = Path(args.figures_dir).expanduser().resolve() if args.figures_dir else project_root / "Part1" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Project root : {project_root}")
    print(f"[INFO] Results dir  : {base_dir}")
    print(f"[INFO] Figures dir  : {figures_dir}")

    for row in print_availability_table(base_dir):
        status = "OK" if row["available"] else "NO RESULTS"
        extra = f" | selected={row['selected_key']}" if row["selected_key"] else ""
        print(f"[{status}] {row['folder']:<35} | {row['dataset']:<7} | {row['init']:<6} | {row['frames']:<8} | {row['optimizer']:<7} | {row['note']}{extra}")

    figure_jobs = [
        ("init", "3dgs", "01_init_compare_3dgs"),
        ("init", "wavelet", "02_init_compare_wavelet"),
        ("optimizer", "colmap", "03_optimizer_compare_colmap"),
        ("optimizer", "vggt", "04_optimizer_compare_vggt"),
        ("frames", None, "05_full_vs_96_colmap_3dgs"),
    ]

    for mode, value, name in figure_jobs:
        fig, _ = make_figure(base_dir, compare_mode=mode, compare_value=value)
        out_path = figures_dir / f"{name}.{args.format}"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[SAVED] {out_path}")
        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
