#!/usr/bin/env python3
"""
Generate Part 1 per-view visualization figures.

This script is converted from part1_per_view_visualization.ipynb and is meant to
be placed under:
    CV-Project4/Part1/scripts/

Default paths, when run from scripts/:
    results input : CV-Project4/Part1/output/
    figures output: CV-Project4/Part1/figures/

Usage:
    python part1_per_view_visualization.py
    python part1_per_view_visualization.py --show
    python part1_per_view_visualization.py --project-root ~/CV-Project4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

METRICS = ["SSIM", "PSNR", "LPIPS"]
DATASETS = ["405841", "dl3dv2", "re10k1"]
INITS = ["colmap", "vggt"]

FOLDERS = {
    ("405841", "colmap", "3dgs"): "405841_colmap_96frames",
    ("405841", "colmap", "wavelet"): "405841_colmap_96frames_wavelet",
    ("405841", "vggt", "3dgs"): "405841_vggt_96frames",
    ("405841", "vggt", "wavelet"): "405841_vggt_96frames_wavelet",
    ("dl3dv2", "colmap", "3dgs"): "dl3dv2_colmap_96frames",
    ("dl3dv2", "colmap", "wavelet"): "dl3dv2_colmap_96frames_wavelet",
    ("dl3dv2", "vggt", "3dgs"): "dl3dv2_vggt_96frames",
    ("dl3dv2", "vggt", "wavelet"): "dl3dv2_vggt_96frames_wavelet",
    ("re10k1", "colmap", "3dgs"): "re10k1_colmap_96frames",
    ("re10k1", "colmap", "wavelet"): "re10k1_colmap_96frames_wavelet",
    ("re10k1", "vggt", "3dgs"): "re10k1_vggt_96frames",
    ("re10k1", "vggt", "wavelet"): "re10k1_vggt_96frames_wavelet",
}

COLORS = {
    "3dgs": "#4C78A8",
    "wavelet": "#83B5D5",
}


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
    parser = argparse.ArgumentParser(description="Generate Part 1 per-view 3DGS vs Wavelet-GS figures.")
    parser.add_argument("--project-root", type=str, default=None, help="Path to CV-Project4. Auto-detected by default.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory containing result folders. Default: <project-root>/Part1/output.")
    parser.add_argument("--figures-dir", type=str, default=None, help="Directory to save figures. Default: <project-root>/Part1/figures.")
    parser.add_argument("--show", action="store_true", help="Also open figures interactively if supported.")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png", help="Output figure format.")
    return parser.parse_args()


def get_folder(dataset: str, init_method: str, optimizer: str) -> Optional[str]:
    return FOLDERS.get((dataset, init_method, optimizer))


def _extract_step_from_key(key: str) -> Optional[int]:
    match = re.fullmatch(r"ours_(\d+)", str(key))
    return int(match.group(1)) if match else None


def _select_valid_block(raw: object) -> Tuple[Optional[Tuple[int, str, dict]], Optional[str]]:
    if not isinstance(raw, dict):
        return None, "json content is not a dict"

    candidates = []
    for key, value in raw.items():
        step = _extract_step_from_key(str(key))
        if step is None or not isinstance(value, dict):
            continue
        valid = all(metric in value and isinstance(value[metric], dict) for metric in METRICS)
        if valid:
            candidates.append((step, str(key), value))

    if not candidates:
        return None, f"no valid ours_* block found; keys={list(raw.keys())}"

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0], None


def load_per_view_json(base_dir: Path, folder_name: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
    if folder_name is None:
        return None, "folder mapping missing"

    folder_path = base_dir / folder_name
    json_path = folder_path / "per_view.json"

    if not folder_path.exists():
        return None, f"folder not found: {folder_path}"
    if not json_path.exists():
        return None, f"per_view.json not found: {json_path}"

    try:
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        selected, err = _select_valid_block(raw)
        if selected is None:
            return None, err
        _, selected_key, block = selected
        return {"selected_key": selected_key, "metrics": block}, None
    except Exception as exc:
        return None, f"failed to read {json_path}: {exc}"


def _png_sort_key(name: str) -> Tuple[int, object, str]:
    s = str(name)
    match = re.search(r"(\d+)", s)
    if match:
        return (0, int(match.group(1)), s)
    return (1, s, s)


def get_common_pngs(data_a: Optional[dict], data_b: Optional[dict]) -> List[str]:
    if data_a is None or data_b is None:
        return []

    shared = None
    for metric in METRICS:
        keys_a = set(data_a["metrics"].get(metric, {}).keys())
        keys_b = set(data_b["metrics"].get(metric, {}).keys())
        metric_shared = keys_a & keys_b
        shared = metric_shared if shared is None else (shared & metric_shared)

    return sorted(shared or [], key=_png_sort_key)


def build_pair_data(base_dir: Path, dataset: str, init_method: str) -> dict:
    folder_3dgs = get_folder(dataset, init_method, "3dgs")
    folder_wavelet = get_folder(dataset, init_method, "wavelet")
    data_3dgs, err_3dgs = load_per_view_json(base_dir, folder_3dgs)
    data_wavelet, err_wavelet = load_per_view_json(base_dir, folder_wavelet)

    return {
        "dataset": dataset,
        "init": init_method,
        "folder_3dgs": folder_3dgs,
        "folder_wavelet": folder_wavelet,
        "data_3dgs": data_3dgs,
        "data_wavelet": data_wavelet,
        "err_3dgs": err_3dgs,
        "err_wavelet": err_wavelet,
        "common_pngs": get_common_pngs(data_3dgs, data_wavelet),
    }


def print_pair_availability(base_dir: Path) -> List[dict]:
    rows = []
    for dataset in DATASETS:
        for init_method in INITS:
            pair = build_pair_data(base_dir, dataset, init_method)
            rows.append({
                "dataset": dataset,
                "init": init_method,
                "folder_3dgs": pair["folder_3dgs"],
                "folder_wavelet": pair["folder_wavelet"],
                "3dgs_ok": pair["data_3dgs"] is not None,
                "wavelet_ok": pair["data_wavelet"] is not None,
                "common_png_count": len(pair["common_pngs"]),
                "3dgs_key": None if pair["data_3dgs"] is None else pair["data_3dgs"]["selected_key"],
                "wavelet_key": None if pair["data_wavelet"] is None else pair["data_wavelet"]["selected_key"],
                "note_3dgs": "ok" if pair["data_3dgs"] is not None else pair["err_3dgs"],
                "note_wavelet": "ok" if pair["data_wavelet"] is not None else pair["err_wavelet"],
            })
    return rows


def _prettify_dataset_name(name: str) -> str:
    return {"405841": "Waymo-405841", "dl3dv2": "DL3DV-2", "re10k1": "Re10K-1"}.get(name, name)


def _prettify_init_name(init_method: str) -> str:
    return "COLMAP" if init_method == "colmap" else "VGGT"


def _metric_ylim(values: Iterable[float], metric: str) -> Tuple[float, float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return 0, 1
    vmin, vmax = min(valid), max(valid)
    if metric == "SSIM":
        lower, upper = max(0.0, vmin - 0.03), min(1.0, vmax + 0.03)
    elif metric == "LPIPS":
        lower, upper = max(0.0, vmin - 0.03), vmax + 0.03
    else:
        pad = max(0.5, 0.08 * (vmax - vmin if vmax > vmin else 1.0))
        lower, upper = max(0.0, vmin - pad), vmax + pad
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def make_per_view_figure(base_dir: Path, dataset: str, init_method: str, save_path: Optional[Path] = None):
    pair = build_pair_data(base_dir, dataset, init_method)
    dataset_name = _prettify_dataset_name(dataset)
    init_name = _prettify_init_name(init_method)
    title = f"{dataset_name} | {init_name} Init | 3DGS vs Wavelet-GS (Per-View)"

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), dpi=300)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.04)

    common_pngs = pair["common_pngs"]
    if len(common_pngs) == 0:
        for ax, metric in zip(axes, METRICS):
            ax.text(0.5, 0.5, "no shared per-view results", ha="center", va="center", fontsize=12, color="#666666", transform=ax.transAxes)
            ax.set_title(metric, fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_alpha(0.25)

        note_lines = []
        if pair["err_3dgs"] is not None:
            note_lines.append(f"3DGS: {pair['err_3dgs']}")
        if pair["err_wavelet"] is not None:
            note_lines.append(f"Wavelet-GS: {pair['err_wavelet']}")
        if note_lines:
            fig.text(0.5, -0.02, " | ".join(note_lines), ha="center", va="top", fontsize=9, color="#666666")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        return fig, axes

    x = np.arange(len(common_pngs))
    x_labels = [Path(p).stem for p in common_pngs]
    legend_handles = [
        Line2D([0], [0], color=COLORS["3dgs"], linewidth=2.2, marker="o", markersize=4, label="3DGS"),
        Line2D([0], [0], color=COLORS["wavelet"], linewidth=2.2, marker="o", markersize=4, label="Wavelet-GS"),
    ]

    for ax, metric in zip(axes, METRICS):
        vals_3dgs = [float(pair["data_3dgs"]["metrics"][metric][png]) for png in common_pngs]
        vals_wavelet = [float(pair["data_wavelet"]["metrics"][metric][png]) for png in common_pngs]

        ax.plot(x, vals_3dgs, color=COLORS["3dgs"], linewidth=2.0, marker="o", markersize=3.8, label="3DGS")
        ax.plot(x, vals_wavelet, color=COLORS["wavelet"], linewidth=2.0, marker="o", markersize=3.8, label="Wavelet-GS")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Image Index", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.28)
        ax.set_axisbelow(True)
        ax.set_ylim(*_metric_ylim(vals_3dgs + vals_wavelet, metric))
        ax.legend(handles=legend_handles, fontsize=9, frameon=False, loc="best")

    footer = (
        f"Shared images used: {len(common_pngs)} | "
        f"3DGS key: {pair['data_3dgs']['selected_key']} | "
        f"Wavelet-GS key: {pair['data_wavelet']['selected_key']}"
    )
    fig.text(0.5, -0.02, footer, ha="center", va="top", fontsize=9, color="#555555")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
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

    for row in print_pair_availability(base_dir):
        print(
            f"[{'OK' if row['3dgs_ok'] else 'NO'} / {'OK' if row['wavelet_ok'] else 'NO'}] "
            f"{row['dataset']:<7} {row['init']:<6} shared={row['common_png_count']:<4} | "
            f"3DGS: {row['note_3dgs']} | Wavelet: {row['note_wavelet']}"
        )

    figure_specs = [
        ("405841", "colmap"), ("405841", "vggt"),
        ("dl3dv2", "colmap"), ("dl3dv2", "vggt"),
        ("re10k1", "colmap"), ("re10k1", "vggt"),
    ]

    for dataset, init_method in figure_specs:
        out_path = figures_dir / f"{dataset}_{init_method}_per_view_3dgs_vs_wavelet.{args.format}"
        fig, _ = make_per_view_figure(base_dir, dataset, init_method, save_path=out_path)
        print(f"[SAVED] {out_path}")
        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
