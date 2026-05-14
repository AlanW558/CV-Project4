#!/usr/bin/env python3
"""
Visualize Part 2 S3PO-GS pipeline refinement results.

This script reads three pipeline_refinement result folders manually provided
from the command line. Each result folder should contain:

    psnr/before_opt/final_result.json
    psnr/after_opt/final_result.json

Expected final_result.json format:

{
    "mean_psnr": 25.36307409193632,
    "mean_ssim": 0.9104701805204042,
    "mean_lpips": 0.03650485301453076
}

Default output directory:

    CV-Project4/Part2/results/fig/

Typical usage from CV-Project4/Part2/scripts:

python part2_pipeline_refinement_visualization.py \
  --waymo405841 ../results/405841/pipeline_refinement/datasets_405841/2026-04-12-00-39-30 \
  --dl3dv2 ../results/DL3DV-2/pipeline_refinement/CV_PROJECT_datasets/2026-04-12-00-36-47 \
  --re10k1 ../results/Re10k-1/pipeline_refinement/CV_PROJECT_datasets/2026-04-12-00-46-36
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


METRICS = ["PSNR", "SSIM", "LPIPS"]
JSON_KEY_MAP = {
    "PSNR": "mean_psnr",
    "SSIM": "mean_ssim",
    "LPIPS": "mean_lpips",
}

COLORS = {
    "before_opt": "#4C78A8",
    "after_opt": "#83B5D5",
}

DATASET_LABELS = {
    "waymo405841": "Waymo-405841",
    "dl3dv2": "DL3DV-2",
    "re10k1": "Re10K-1",
}


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find CV-Project4 root by walking upward from start/current file."""
    if start is None:
        start = Path(__file__).resolve().parent
    start = start.resolve()

    candidates = [start] + list(start.parents)
    for candidate in candidates:
        if candidate.name == "CV-Project4" and (candidate / "Part2").exists():
            return candidate
        if (candidate / "Part2").exists() and (candidate / "datasets").exists():
            return candidate

    # Reasonable fallback when the script is placed in CV-Project4/Part2/scripts.
    if start.name == "scripts" and start.parent.name == "Part2":
        return start.parent.parent

    raise FileNotFoundError(
        "Could not locate CV-Project4 root. Please pass --project-root explicitly."
    )


def get_json_path(pipeline_refinement_path: Path, stage: str) -> Path:
    return pipeline_refinement_path / "psnr" / stage / "final_result.json"


def load_final_result(json_path: Path) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    if not json_path.exists():
        return None, f"final_result.json not found: {json_path}"

    try:
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        result = {}
        for metric, json_key in JSON_KEY_MAP.items():
            if json_key not in raw:
                return None, f"missing key {json_key!r} in {json_path}"
            result[metric] = float(raw[json_key])

        return result, None
    except Exception as exc:  # noqa: BLE001
        return None, f"failed to read {json_path}: {exc}"


def collect_results(dataset_paths: Dict[str, Path]) -> Dict[str, Dict[str, object]]:
    collected: Dict[str, Dict[str, object]] = {}
    for dataset_name, pipeline_path in dataset_paths.items():
        before_path = get_json_path(pipeline_path, "before_opt")
        after_path = get_json_path(pipeline_path, "after_opt")

        before_result, before_err = load_final_result(before_path)
        after_result, after_err = load_final_result(after_path)

        collected[dataset_name] = {
            "before_opt": before_result,
            "after_opt": after_result,
            "before_err": before_err,
            "after_err": after_err,
            "before_path": str(before_path),
            "after_path": str(after_path),
        }
    return collected


def print_availability(collected: Dict[str, Dict[str, object]]) -> None:
    print("\nResult availability:")
    for dataset_name, item in collected.items():
        before_note = "ok" if item["before_opt"] is not None else item["before_err"]
        after_note = "ok" if item["after_opt"] is not None else item["after_err"]
        print(f"  [{dataset_name}] before_opt: {before_note}")
        print(f"  [{dataset_name}] after_opt : {after_note}")


def make_pipeline_refinement_figure(
    dataset_paths: Dict[str, Path],
    save_path: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    collected = collect_results(dataset_paths)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=300)
    fig.suptitle(
        "Part 2 | S3PO-GS Pipeline Refinement | Before vs After Color Refinement",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    dataset_names = list(dataset_paths.keys())
    metric_positions = np.arange(len(dataset_names))
    width = 0.32

    specs = [
        {"label": "Before Opt", "key": "before_opt", "color": COLORS["before_opt"]},
        {"label": "After Opt", "key": "after_opt", "color": COLORS["after_opt"]},
    ]

    legend_handles = [
        Patch(facecolor=s["color"], edgecolor="white", label=s["label"])
        for s in specs
    ]

    for ax, metric in zip(axes, METRICS):
        all_valid_vals = []

        for i, spec in enumerate(specs):
            offsets = metric_positions + (i - (len(specs) - 1) / 2) * width
            heights = []
            raw_vals = []

            for dataset_name in dataset_names:
                result = collected[dataset_name][spec["key"]]
                if result is None:
                    heights.append(0.0)
                    raw_vals.append(None)
                else:
                    val = result[metric]
                    heights.append(val)
                    raw_vals.append(val)
                    all_valid_vals.append(val)

            bars = ax.bar(
                offsets,
                heights,
                width=width * 0.92,
                color=spec["color"],
                edgecolor="white",
                linewidth=0.9,
                label=spec["label"],
            )

            for bar, val in zip(bars, raw_vals):
                x = bar.get_x() + bar.get_width() / 2
                if val is None:
                    bar.set_alpha(0.25)
                    bar.set_hatch("//")
                else:
                    ax.text(
                        x,
                        bar.get_height() + 0.01,
                        f"{val:.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        if all_valid_vals:
            ymin = 0.0
            ymax = max(all_valid_vals) * 1.22
            if metric == "SSIM":
                ymax = min(1.0, max(ymax, max(all_valid_vals) + 0.05))
            elif metric == "LPIPS":
                ymax = max(ymax, max(all_valid_vals) + 0.02)
            else:
                ymax = max(ymax, max(all_valid_vals) + 0.5)
        else:
            ymin, ymax = 0.0, 1.0

        ax.set_ylim(ymin, ymax)

        for i, spec in enumerate(specs):
            offsets = metric_positions + (i - (len(specs) - 1) / 2) * width
            for x, dataset_name in zip(offsets, dataset_names):
                result = collected[dataset_name][spec["key"]]
                if result is None:
                    ax.text(
                        x,
                        ymax * 0.03,
                        "no results",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=8,
                        color="#666666",
                    )

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(metric_positions)
        ax.set_xticklabels(dataset_names, fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(handles=legend_handles, fontsize=9, frameon=False, loc="upper left")

    footer_items = []
    for dataset_name in dataset_names:
        before_ok = collected[dataset_name]["before_opt"] is not None
        after_ok = collected[dataset_name]["after_opt"] is not None
        footer_items.append(
            f"{dataset_name}: before={'ok' if before_ok else 'missing'}, "
            f"after={'ok' if after_ok else 'missing'}"
        )

    fig.text(
        0.5,
        -0.02,
        " | ".join(footer_items),
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Part 2 pipeline refinement results for three datasets."
    )
    parser.add_argument(
        "--waymo405841",
        required=True,
        type=Path,
        help="Path to Waymo-405841 pipeline_refinement run directory.",
    )
    parser.add_argument(
        "--dl3dv2",
        required=True,
        type=Path,
        help="Path to DL3DV-2 pipeline_refinement run directory.",
    )
    parser.add_argument(
        "--re10k1",
        required=True,
        type=Path,
        help="Path to Re10k-1 pipeline_refinement run directory.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Path to CV-Project4. If omitted, the script tries to infer it.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="part2_pipeline_refinement_comparison.png",
        help="Output figure filename saved under CV-Project4/Part2/results/fig/.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve() if args.project_root else find_project_root()
    output_dir = project_root / "Part2" / "results" / "fig"
    save_path = output_dir / args.output_name

    dataset_paths = {
        DATASET_LABELS["waymo405841"]: args.waymo405841.expanduser().resolve(),
        DATASET_LABELS["dl3dv2"]: args.dl3dv2.expanduser().resolve(),
        DATASET_LABELS["re10k1"]: args.re10k1.expanduser().resolve(),
    }

    collected = collect_results(dataset_paths)
    print_availability(collected)

    fig, _ = make_pipeline_refinement_figure(dataset_paths, save_path=save_path)
    print(f"\nSaved figure to: {save_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
