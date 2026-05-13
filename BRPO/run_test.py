#!/usr/bin/env python3
"""
Standalone held-out test evaluator for S3PO-GS.

What it does
------------
1. Loads a held-out test dataset from a yaml config.
2. Loads a trained Gaussian point cloud from a .ply file.
3. Builds test cameras from the dataset.
4. Renders every test frame with either:
   - GT poses (`--pose_mode gt`), or
   - sequentially inferred poses with MASt3R (`--pose_mode infer`)
5. Computes PSNR / SSIM / LPIPS and saves results.

Recommended usage
-----------------
python run_test.py \
  --config /path/to/test.yaml \
  --test_path /path/to/test_split \
  --ply_path /path/to/point_cloud.ply \
  --save_dir /path/to/output \
  --pose_mode gt \
  --origin_mode auto

Notes
-----
- The yaml config is still required because the repo's dataset loader expects
  Dataset.type, Dataset.Calibration, model_params, and pipeline_params.
- `--test_path` overrides Dataset.dataset_path in the yaml.
- For dl3dv/re10k, begin/end are required by the parser; this script sets them automatically
  if they are missing.
- `origin_mode auto` tries to align test and sparse origins for directory layouts like:
      scene/test  <->  scene/sparse
  or:
      scene_test  <->  scene_sparse
  For Waymo, `none` is often safest.
"""

import argparse
import copy
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from munch import munchify
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p

from utils.camera_utils import Camera
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import evaluate_evo
from utils.init_pose import get_pose
from utils.logging_utils import Log
from mast3r.model import AsymmetricMASt3R


def parse_args():
    parser = argparse.ArgumentParser("Standalone held-out test evaluator")
    parser.add_argument("--config", type=str, required=True, help="Path to test yaml config")
    parser.add_argument("--test_path", type=str, required=True, help="Held-out test dataset path")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to trained Gaussian .ply")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--pose_mode", type=str, default="gt", choices=["gt", "infer"])
    parser.add_argument("--origin_mode", type=str, default="auto",
                        choices=["none", "auto", "scene_test_sparse", "scene_test_folder"])
    parser.add_argument("--begin", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--mast3r_model_name", type=str,
                        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument("--infer_init_mode", type=str, default="gt_first",
                        choices=["gt_first", "identity"])
    parser.add_argument("--save_render_rgb", action="store_true", default=True)
    parser.add_argument("--save_render_depth", action="store_true")
    parser.add_argument("--save_render_depth_npy", action="store_true")
    return parser.parse_args()


def count_images_for_parser(dataset_type: str, dataset_path: Path) -> int:
    if dataset_type in ["dl3dv", "KITTI"]:
        return len(sorted(glob.glob(str(dataset_path / "rgb" / "*.png"))))
    if dataset_type == "waymo":
        return len(sorted(glob.glob(str(dataset_path / "rgb" / "*.png"))))
    return len(sorted(glob.glob(str(dataset_path / "rgb" / "*.png"))))


def build_config(args):
    config = load_config(args.config)
    config = copy.deepcopy(config)
    config["Dataset"]["dataset_path"] = args.test_path

    # The repo parser for dl3dv/KITTI expects begin/end to exist.
    if args.begin is not None:
        config["Dataset"]["begin"] = args.begin
    else:
        config["Dataset"]["begin"] = config["Dataset"].get("begin", 0)

    if args.end is not None:
        config["Dataset"]["end"] = args.end
    else:
        n = count_images_for_parser(config["Dataset"]["type"], Path(args.test_path))
        config["Dataset"]["end"] = n

    return config


def load_gaussians_from_ply(config, ply_path):
    sh_degree = int(config["model_params"].get("sh_degree", 0))
    gaussians = GaussianModel(sh_degree, config=config)
    gaussians.load_ply(ply_path)
    return gaussians


def build_projection_matrix(dataset, device="cuda"):
    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=dataset.fx,
        fy=dataset.fy,
        cx=dataset.cx,
        cy=dataset.cy,
        W=dataset.width,
        H=dataset.height,
    ).transpose(0, 1)
    return projection_matrix.to(device=device)


def _read_first_translation(cameras_json_path: Path):
    if not cameras_json_path.exists():
        return None
    with open(cameras_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return None
    return np.asarray(data[0]["cam_trans"], dtype=np.float64)


def compute_world_offset(test_path: Path, origin_mode: str):
    """Return world offset to map test local origin to sparse local origin.

    Supports both:
      scene/test   <-> scene/sparse
      scene_test   <-> scene_sparse
    """
    if origin_mode == "none":
        return None

    candidates = []
    if origin_mode in ["auto", "scene_test_folder"]:
        # scene/test -> scene/sparse
        candidates.append(test_path.parent / "sparse" / "cameras.json")
    if origin_mode in ["auto", "scene_test_sparse"]:
        # scene_test -> scene_sparse
        if test_path.name.endswith("_test"):
            candidates.append(test_path.parent / test_path.name.replace("_test", "_sparse") / "cameras.json")

    test_cameras = test_path / "cameras.json"
    test_first = _read_first_translation(test_cameras)
    if test_first is None:
        return None

    for sparse_cameras in candidates:
        sparse_first = _read_first_translation(sparse_cameras)
        if sparse_first is not None:
            return test_first - sparse_first

    return None


def shift_w2c_by_world_offset(pose_w2c, world_offset):
    shifted = pose_w2c.copy()
    shifted[:3, 3] = pose_w2c[:3, 3] - pose_w2c[:3, :3] @ world_offset
    return shifted


def build_cameras_from_dataset(dataset, projection_matrix, config, world_offset=None):
    cameras = []
    for idx in range(len(dataset)):
        cam = Camera.init_from_dataset(dataset, idx, projection_matrix)
        cam.compute_grad_mask(config)

        if world_offset is not None:
            pose_w2c = np.eye(4, dtype=np.float64)
            pose_w2c[:3, :3] = cam.R_gt.detach().cpu().numpy()
            pose_w2c[:3, 3] = cam.T_gt.detach().cpu().numpy()
            pose_w2c = shift_w2c_by_world_offset(pose_w2c, world_offset)
            cam.R_gt = torch.tensor(pose_w2c[:3, :3], device=cam.device, dtype=torch.float32)
            cam.T_gt = torch.tensor(pose_w2c[:3, 3], device=cam.device, dtype=torch.float32)

        cameras.append(cam)
    return cameras


def apply_gt_pose_to_cameras(cameras):
    for cam in cameras:
        cam.update_RT(cam.R_gt.clone(), cam.T_gt.clone())


def camera_w2c_matrix(camera, use_gt=False):
    pose = np.eye(4, dtype=np.float64)
    if use_gt:
        pose[:3, :3] = camera.R_gt.detach().cpu().numpy()
        pose[:3, 3] = camera.T_gt.detach().cpu().numpy()
    else:
        pose[:3, :3] = camera.R.detach().cpu().numpy()
        pose[:3, 3] = camera.T.detach().cpu().numpy()
    return pose


def update_camera_pose(camera, pose_w2c):
    camera.update_RT(
        torch.tensor(pose_w2c[:3, :3], device=camera.device, dtype=torch.float32),
        torch.tensor(pose_w2c[:3, 3], device=camera.device, dtype=torch.float32),
    )


def infer_poses_sequential(
    cameras, dataset, gaussians, pipeline_params, background, mast3r_model, dist_coeffs, init_mode="gt_first"
):
    if len(cameras) == 0:
        return [], []

    estimated_poses_w2c = []
    pose_success = []

    init_pose = np.eye(4, dtype=np.float64) if init_mode == "identity" else camera_w2c_matrix(cameras[0], use_gt=True)
    update_camera_pose(cameras[0], init_pose)
    estimated_poses_w2c.append(init_pose)
    pose_success.append(True)

    for idx in range(1, len(cameras)):
        ref_idx = idx - 1
        ref_cam = cameras[ref_idx]

        img_ref, _, _, _ = dataset[ref_idx]
        img_cur, _, _, _ = dataset[idx]

        rel_pose, _ = get_pose(
            img1=img_ref,
            img2=img_cur,
            model=mast3r_model,
            dist_coeffs=dist_coeffs,
            viewpoint=ref_cam,
            gaussians=gaussians,
            pipeline_params=pipeline_params,
            background=background,
        )

        rel_pose = np.asarray(rel_pose, dtype=np.float64)
        if rel_pose.shape != (4, 4):
            rel_pose = np.eye(4, dtype=np.float64)

        ref_pose = camera_w2c_matrix(ref_cam, use_gt=False)
        is_valid = not np.allclose(rel_pose, np.eye(4), atol=1e-6)
        cur_pose = rel_pose @ ref_pose if is_valid else ref_pose.copy()

        update_camera_pose(cameras[idx], cur_pose)
        estimated_poses_w2c.append(cur_pose)
        pose_success.append(bool(is_valid))

    return estimated_poses_w2c, pose_success


def evaluate_external_ate(gt_poses_w2c, est_poses_w2c, save_dir, label="external_infer", monocular=True):
    gt_poses_c2w = [np.linalg.inv(pose) for pose in gt_poses_w2c]
    est_poses_c2w = [np.linalg.inv(pose) for pose in est_poses_w2c]

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    trj_path = os.path.join(plot_dir, f"trj_{label}.json")
    with open(trj_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trj_id": list(range(len(gt_poses_c2w))),
                "trj_gt": [pose.tolist() for pose in gt_poses_c2w],
                "trj_est": [pose.tolist() for pose in est_poses_c2w],
            },
            f,
            indent=4,
        )

    ate_rmse = evaluate_evo(
        poses_gt=gt_poses_c2w,
        poses_est=est_poses_c2w,
        plot_dir=plot_dir,
        label=label,
        monocular=monocular,
    )
    return {"ate_rmse": float(ate_rmse), "traj_file": trj_path, "label": label}


def _prepare_output_dirs(save_dir, save_render_rgb, save_render_depth, save_render_depth_npy):
    output_dirs = {}
    mkdir_p(save_dir)
    if save_render_rgb:
        output_dirs["render_rgb"] = os.path.join(save_dir, "render_rgb")
        mkdir_p(output_dirs["render_rgb"])
    if save_render_depth:
        output_dirs["render_depth"] = os.path.join(save_dir, "render_depth")
        mkdir_p(output_dirs["render_depth"])
    if save_render_depth_npy:
        output_dirs["render_depth_npy"] = os.path.join(save_dir, "render_depth_npy")
        mkdir_p(output_dirs["render_depth_npy"])
    return output_dirs


def _to_uint8_rgb(image_tensor):
    return (image_tensor.detach().cpu().numpy().transpose((1, 2, 0)).clip(0.0, 1.0) * 255.0).astype(np.uint8)


def _save_depth_outputs(depth_tensor, frame_idx, output_dirs):
    depth_np = depth_tensor.detach().cpu().numpy()
    if "render_depth_npy" in output_dirs:
        np.save(os.path.join(output_dirs["render_depth_npy"], f"{frame_idx:04d}.npy"), depth_np)

    if "render_depth" in output_dirs:
        depth_min = float(depth_np.min())
        depth_max = float(depth_np.max())
        if depth_max > depth_min:
            depth_norm = (depth_np - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_np)
        Image.fromarray((depth_norm * 255).astype(np.uint8)).save(
            os.path.join(output_dirs["render_depth"], f"{frame_idx:04d}.png")
        )


def run_render_eval(cameras, dataset, gaussians, pipe, background, save_dir,
                    pose_mode, save_render_rgb=True, save_render_depth=False, save_render_depth_npy=False):
    output_dirs = _prepare_output_dirs(save_dir, save_render_rgb, save_render_depth, save_render_depth_npy)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")

    frame_metrics, psnr_values, ssim_values, lpips_values = [], [], [], []

    for idx, camera in enumerate(cameras):
        gt_image, _, _, _ = dataset[idx]
        render_pkg = render(camera, gaussians, pipe, background)
        pred_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        depth_map = render_pkg["depth"].squeeze()

        valid_mask = gt_image > 0
        if torch.count_nonzero(valid_mask) > 0:
            psnr_score = psnr(pred_image[valid_mask].unsqueeze(0), gt_image[valid_mask].unsqueeze(0))
        else:
            psnr_score = psnr(pred_image.unsqueeze(0), gt_image.unsqueeze(0))

        ssim_score = ssim(pred_image.unsqueeze(0), gt_image.unsqueeze(0))
        lpips_score = lpips_metric(pred_image.unsqueeze(0), gt_image.unsqueeze(0))

        psnr_value = float(psnr_score.item())
        ssim_value = float(ssim_score.item())
        lpips_value = float(lpips_score.item())

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)

        frame_metrics.append({
            "frame_idx": idx,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value,
        })

        if "render_rgb" in output_dirs:
            Image.fromarray(_to_uint8_rgb(pred_image)).save(
                os.path.join(output_dirs["render_rgb"], f"{idx:04d}_pred.png")
            )

        if "render_depth" in output_dirs or "render_depth_npy" in output_dirs:
            _save_depth_outputs(depth_map, idx, output_dirs)

    summary = {
        "pose_mode": pose_mode,
        "num_frames": len(frame_metrics),
        "avg_psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
        "avg_ssim": float(np.mean(ssim_values)) if ssim_values else 0.0,
        "avg_lpips": float(np.mean(lpips_values)) if lpips_values else 0.0,
        "metrics": frame_metrics,
    }
    return summary


def write_results(save_dir, result):
    mkdir_p(save_dir)
    full_result_path = os.path.join(save_dir, "eval_heldout.json")
    with open(full_result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    compact = {
        "num_frames": result["num_frames"],
        "avg_psnr": result["avg_psnr"],
        "avg_ssim": result["avg_ssim"],
        "avg_lpips": result["avg_lpips"],
    }
    compact_dir = os.path.join(save_dir, "psnr", str(result["pose_mode"]))
    mkdir_p(compact_dir)
    with open(os.path.join(compact_dir, "final_result.json"), "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=4)

    return full_result_path


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    config = build_config(args)
    save_dir = args.save_dir
    mkdir_p(save_dir)

    with open(os.path.join(save_dir, "config_heldout_eval.yml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    model_params = munchify(config["model_params"])
    pipe_params = munchify(config["pipeline_params"])

    dataset = load_dataset(model_params, model_params.source_path, config=config)
    projection_matrix = build_projection_matrix(dataset, device="cuda")

    world_offset = compute_world_offset(Path(args.test_path), args.origin_mode)
    if world_offset is not None:
        Log(f"Using world offset: {world_offset.tolist()}", tag="Eval")

    cameras = build_cameras_from_dataset(dataset, projection_matrix, config, world_offset=world_offset)

    gaussians = load_gaussians_from_ply(config, args.ply_path)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    pose_result = None
    if args.pose_mode == "gt":
        apply_gt_pose_to_cameras(cameras)
    else:
        mast3r_model = AsymmetricMASt3R.from_pretrained(args.mast3r_model_name).to("cuda")
        est_poses_w2c, pose_success = infer_poses_sequential(
            cameras=cameras,
            dataset=dataset,
            gaussians=gaussians,
            pipeline_params=pipe_params,
            background=background,
            mast3r_model=mast3r_model,
            dist_coeffs=dataset.dist_coeffs,
            init_mode=args.infer_init_mode,
        )
        gt_poses_w2c = [camera_w2c_matrix(cam, use_gt=True) for cam in cameras]
        pose_result = evaluate_external_ate(
            gt_poses_w2c=gt_poses_w2c,
            est_poses_w2c=est_poses_w2c,
            save_dir=save_dir,
            label="heldout_infer",
            monocular=config["Dataset"]["sensor_type"] == "monocular",
        )
        pose_result["pose_success_rate"] = float(np.mean(pose_success)) if pose_success else 0.0
        pose_result["pose_success_count"] = int(sum(pose_success))
        pose_result["pose_total_count"] = len(pose_success)

    render_result = run_render_eval(
        cameras=cameras,
        dataset=dataset,
        gaussians=gaussians,
        pipe=pipe_params,
        background=background,
        save_dir=save_dir,
        pose_mode=args.pose_mode,
        save_render_rgb=args.save_render_rgb,
        save_render_depth=args.save_render_depth,
        save_render_depth_npy=args.save_render_depth_npy,
    )

    result = dict(render_result)
    result["config"] = args.config
    result["test_path"] = args.test_path
    result["ply_path"] = args.ply_path
    result["save_dir"] = save_dir
    result["origin_mode"] = args.origin_mode
    result["world_offset"] = world_offset.tolist() if world_offset is not None else None

    if pose_result is not None:
        result["pose"] = pose_result

    result_path = write_results(save_dir, result)
    Log(f"Held-out test evaluation complete: {result_path}", tag="Eval")


if __name__ == "__main__":
    main()
