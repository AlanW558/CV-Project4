import os
import json
import copy
import torch
import numpy as np
from PIL import Image

from gaussian_splatting.gaussian_renderer import render


def save_tensor_image(image_tensor, save_path):
    image = torch.clamp(image_tensor.detach(), 0.0, 1.0)
    image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(image_np).save(save_path)


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_float_array(x, path):
    x_np = _to_numpy(x)
    if x_np is None:
        return None
    np.save(path, x_np.astype(np.float32))
    return path


def save_depth_vis(depth, path):
    depth_np = _to_numpy(depth)
    if depth_np is None:
        return None

    depth_np = np.squeeze(depth_np).astype(np.float32)
    valid = np.isfinite(depth_np) & (depth_np > 0)

    if valid.sum() == 0:
        Image.fromarray(np.zeros_like(depth_np, dtype=np.uint8)).save(path)
        return path

    lo, hi = np.percentile(depth_np[valid], [2, 98])
    depth_vis = np.clip((depth_np - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    Image.fromarray(depth_vis).save(path)
    return path


def w2c_to_c2w(R_w2c, T_w2c):
    R_w2c = R_w2c.detach()
    T_w2c = T_w2c.detach().reshape(3)

    R_c2w = R_w2c.transpose(0, 1)
    C = -R_c2w @ T_w2c
    return R_c2w, C


def c2w_to_w2c(R_c2w, C):
    R_w2c = R_c2w.transpose(0, 1)
    T_w2c = -R_w2c @ C.reshape(3)
    return R_w2c.contiguous(), T_w2c.contiguous()


def project_to_so3(R):
    U, _, Vh = torch.linalg.svd(R)
    R_proj = U @ Vh
    if torch.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vh
    return R_proj


def slerp_rotation(R0, R1, alpha):
    R0 = R0.detach()
    R1 = R1.detach()

    R_rel = project_to_so3(R0.transpose(0, 1) @ R1)
    cos_theta = ((torch.trace(R_rel) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta.abs() < 1e-6:
        return project_to_so3((1.0 - alpha) * R0 + alpha * R1)

    omega_hat = (R_rel - R_rel.transpose(0, 1)) / (2.0 * torch.sin(theta) + 1e-8)
    I = torch.eye(3, dtype=R0.dtype, device=R0.device)

    R_delta = (
        I
        + torch.sin(alpha * theta) * omega_hat
        + (1.0 - torch.cos(alpha * theta)) * (omega_hat @ omega_hat)
    )

    return project_to_so3(R0 @ R_delta)

def get_camera_intrinsics_numpy(cam):
    fx = float(cam.fx)
    fy = float(cam.fy)
    cx = float(cam.cx)
    cy = float(cam.cy)

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K

def get_w2c_numpy(cam):
    R = cam.R.detach().cpu().numpy() if torch.is_tensor(cam.R) else np.asarray(cam.R)
    T = cam.T.detach().cpu().numpy() if torch.is_tensor(cam.T) else np.asarray(cam.T)

    R = R.astype(np.float32)
    T = T.reshape(3).astype(np.float32)

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = T
    return w2c

def warp_depth_to_target(
    src_cam,
    tgt_cam,
    src_depth,
    tgt_h,
    tgt_w,
    depth_scale=1.0,
    eps=1e-6,
):
    """
    Warp source depth map into target camera using z-buffer.

    Returns:
        tgt_depth: HxW depth in target camera coordinates
        tgt_valid: HxW binary mask
    """
    if torch.is_tensor(src_depth):
        d = src_depth.detach().cpu().numpy()
    else:
        d = np.asarray(src_depth)

    d = d.squeeze().astype(np.float32) * float(depth_scale)

    Hs, Ws = d.shape

    Ks = get_camera_intrinsics_numpy(src_cam)
    Kt = get_camera_intrinsics_numpy(tgt_cam)

    src_w2c = get_w2c_numpy(src_cam)
    tgt_w2c = get_w2c_numpy(tgt_cam)

    src_c2w = np.linalg.inv(src_w2c)
    src_to_tgt = tgt_w2c @ src_c2w

    ys, xs = np.meshgrid(
        np.arange(Hs, dtype=np.float32),
        np.arange(Ws, dtype=np.float32),
        indexing="ij",
    )

    z = d.reshape(-1)
    x = xs.reshape(-1)
    y = ys.reshape(-1)

    valid = np.isfinite(z) & (z > eps)
    if valid.sum() == 0:
        return (
            np.zeros((tgt_h, tgt_w), dtype=np.float32),
            np.zeros((tgt_h, tgt_w), dtype=np.float32),
        )

    x = x[valid]
    y = y[valid]
    z = z[valid]

    X = (x - Ks[0, 2]) / Ks[0, 0] * z
    Y = (y - Ks[1, 2]) / Ks[1, 1] * z
    pts_src = np.stack([X, Y, z, np.ones_like(z)], axis=0)

    pts_tgt = src_to_tgt @ pts_src
    Xt = pts_tgt[0]
    Yt = pts_tgt[1]
    Zt = pts_tgt[2]

    valid_z = np.isfinite(Zt) & (Zt > eps)
    Xt = Xt[valid_z]
    Yt = Yt[valid_z]
    Zt = Zt[valid_z]

    u = Kt[0, 0] * (Xt / Zt) + Kt[0, 2]
    v = Kt[1, 1] * (Yt / Zt) + Kt[1, 2]

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    inside = (ui >= 0) & (ui < tgt_w) & (vi >= 0) & (vi < tgt_h)

    ui = ui[inside]
    vi = vi[inside]
    Zt = Zt[inside]

    tgt_depth = np.full((tgt_h, tgt_w), np.inf, dtype=np.float32)

    # z-buffer: keep nearest depth
    flat_idx = vi * tgt_w + ui
    tgt_flat = tgt_depth.reshape(-1)
    np.minimum.at(tgt_flat, flat_idx, Zt)

    tgt_depth = tgt_flat.reshape(tgt_h, tgt_w)
    tgt_valid = np.isfinite(tgt_depth).astype(np.float32)
    tgt_depth[~np.isfinite(tgt_depth)] = 0.0

    return tgt_depth.astype(np.float32), tgt_valid.astype(np.float32)

def align_depth_to_reference_median(depth, ref_depth, valid_mask, eps=1e-6):
    depth = np.asarray(depth).astype(np.float32)
    ref_depth = np.asarray(ref_depth).astype(np.float32)
    valid_mask = np.asarray(valid_mask).astype(bool)

    valid = (
        valid_mask
        & np.isfinite(depth)
        & np.isfinite(ref_depth)
        & (depth > eps)
        & (ref_depth > eps)
    )

    if valid.sum() < 50:
        return depth

    ratio = ref_depth[valid] / (depth[valid] + eps)
    ratio = ratio[np.isfinite(ratio)]

    if ratio.size < 50:
        return depth

    lo, hi = np.percentile(ratio, [10, 90])
    ratio = ratio[(ratio >= lo) & (ratio <= hi)]

    if ratio.size == 0:
        return depth

    scale = float(np.median(ratio))
    return depth * scale

def build_pseudo_depth_by_warping(
    pseudo_cam,
    prev_cam,
    curr_cam,
    prev_depth,
    curr_depth,
    alpha=0.5,
    pseudo_render_depth=None,
    eps=1e-6,
):
    """
    Build geometrically valid pseudo target depth by warping adjacent keyframe depths
    into the pseudo camera.

    If pseudo_render_depth is provided, use it to estimate scale alignment for
    warped depths.
    """
    if pseudo_render_depth is not None:
        if torch.is_tensor(pseudo_render_depth):
            pr = pseudo_render_depth.detach().cpu().numpy().squeeze()
        else:
            pr = np.asarray(pseudo_render_depth).squeeze()
        tgt_h, tgt_w = pr.shape
    else:
        tgt_h = int(pseudo_cam.image_height)
        tgt_w = int(pseudo_cam.image_width)
        pr = None

    d_prev, m_prev = warp_depth_to_target(
        src_cam=prev_cam,
        tgt_cam=pseudo_cam,
        src_depth=prev_depth,
        tgt_h=tgt_h,
        tgt_w=tgt_w,
    )

    d_curr, m_curr = warp_depth_to_target(
        src_cam=curr_cam,
        tgt_cam=pseudo_cam,
        src_depth=curr_depth,
        tgt_h=tgt_h,
        tgt_w=tgt_w,
    )

    # Optional scale alignment to pseudo rendered depth.
    if pr is not None:
        d_prev = align_depth_to_reference_median(d_prev, pr, m_prev)
        d_curr = align_depth_to_reference_median(d_curr, pr, m_curr)

    w_prev = (1.0 - float(alpha)) * m_prev
    w_curr = float(alpha) * m_curr

    denom = w_prev + w_curr + eps
    fused = (w_prev * d_prev + w_curr * d_curr) / denom

    valid = (denom > eps) & np.isfinite(fused) & (fused > eps)
    fused[~valid] = 0.0

    return fused.astype(np.float32), valid.astype(np.float32)

def interpolate_camera_pose(cam_a, cam_b, alpha=0.5):
    """
    Fallback only:
    interpolate in camera-to-world space instead of directly interpolating W2C R/T.
    """
    R0_c2w, C0 = w2c_to_c2w(cam_a.R, cam_a.T)
    R1_c2w, C1 = w2c_to_c2w(cam_b.R, cam_b.T)

    C_mid = (1.0 - alpha) * C0 + alpha * C1
    R_mid_c2w = slerp_rotation(R0_c2w, R1_c2w, alpha)

    R_mid_w2c, T_mid_w2c = c2w_to_w2c(R_mid_c2w, C_mid)
    return R_mid_w2c, T_mid_w2c


def make_pseudo_camera(cam_a, cam_b, alpha=0.5, uid=-1):
    """
    Backward-compatible fallback:
    pseudo pose from keyframe interpolation.
    """
    pseudo_cam = copy.deepcopy(cam_a)
    pseudo_cam.uid = int(uid)

    R_mid, T_mid = interpolate_camera_pose(cam_a, cam_b, alpha)
    pseudo_cam.update_RT(R_mid, T_mid)

    pseudo_cam.original_image = None
    pseudo_cam.depth = None
    pseudo_cam.mono_depth = None
    pseudo_cam.grad_mask = None

    return pseudo_cam


def make_pseudo_camera_from_tracked_frame(tracked_cam, uid=-1):
    """
    Recommended for your current SLAM setting:
    use an already tracked non-keyframe pose as pseudo pose.
    """
    pseudo_cam = copy.deepcopy(tracked_cam)
    pseudo_cam.uid = int(uid)

    pseudo_cam.original_image = None
    pseudo_cam.depth = None
    pseudo_cam.mono_depth = None
    pseudo_cam.grad_mask = None

    return pseudo_cam


def select_tracked_pseudo_frame(cameras, prev_kf_idx, curr_kf_idx, strategy="middle"):
    """
    Select one tracked non-keyframe between two adjacent keyframes.

    cameras: usually self.cameras, dict-like.
    """
    candidates = [
        i for i in range(int(prev_kf_idx) + 1, int(curr_kf_idx))
        if i in cameras
    ]

    if len(candidates) == 0:
        return None

    if strategy == "middle":
        return candidates[len(candidates) // 2]

    if strategy == "first":
        return candidates[0]

    if strategy == "last":
        return candidates[-1]

    return candidates[len(candidates) // 2]

def select_tracked_pseudo_frame_by_alpha(
    cameras,
    prev_kf_idx,
    curr_kf_idx,
    alpha=0.5,
):
    """
    Select tracked non-keyframe according to alpha position.

    alpha=0.25 -> earlier tracked frame
    alpha=0.50 -> middle tracked frame
    alpha=0.75 -> later tracked frame
    """
    candidates = [
        i
        for i in range(int(prev_kf_idx) + 1, int(curr_kf_idx))
        if i in cameras
    ]

    if len(candidates) == 0:
        return None

    alpha = float(np.clip(alpha, 0.0, 1.0))

    pos = int(round(alpha * (len(candidates) - 1)))

    pos = max(0, min(pos, len(candidates) - 1))

    return candidates[pos]


def compute_valid_ratio(depth=None, opacity=None, opacity_threshold=0.05):
    valid = None

    if depth is not None:
        d = np.squeeze(_to_numpy(depth))
        valid_depth = np.isfinite(d) & (d > 0)
        valid = valid_depth if valid is None else (valid & valid_depth)

    if opacity is not None:
        o = np.squeeze(_to_numpy(opacity))
        valid_opacity = np.isfinite(o) & (o > opacity_threshold)

        if valid is not None and valid.shape != valid_opacity.shape:
            return None

        valid = valid_opacity if valid is None else (valid & valid_opacity)

    if valid is None:
        return None

    return float(valid.mean())


@torch.no_grad()
def render_coarse_pseudo_view(
    pseudo_cam,
    gaussians,
    pipe,
    background,
    output_dir,
    prefix,
    prev_kf_idx=None,
    curr_kf_idx=None,
    source_frame_idx=None,
    alpha=0.5,
    pose_mode="tracked",
    min_valid_ratio=0.01,
):
    """
    BRPO-style coarse pseudo-view rendering:
    render I_t^gs under pseudo pose and save RGB/depth/opacity/meta.
    """
    os.makedirs(output_dir, exist_ok=True)

    render_pkg = render(pseudo_cam, gaussians, pipe, background)
    if render_pkg is None or "render" not in render_pkg:
        return None

    rgb = render_pkg["render"]
    depth = render_pkg.get("depth", None)
    opacity = render_pkg.get("opacity", None)
    n_touched = render_pkg.get("n_touched", None)

    valid_ratio = compute_valid_ratio(depth=depth, opacity=opacity)

    if valid_ratio is not None and valid_ratio < min_valid_ratio:
        return None

    image_path = os.path.join(output_dir, f"{prefix}.png")
    depth_path = os.path.join(output_dir, f"{prefix}_depth.npy")
    depth_vis_path = os.path.join(output_dir, f"{prefix}_depth_vis.png")
    opacity_path = os.path.join(output_dir, f"{prefix}_opacity.npy")
    meta_path = os.path.join(output_dir, f"{prefix}_meta.json")

    save_tensor_image(rgb, image_path)

    if depth is not None:
        save_float_array(depth, depth_path)
        save_depth_vis(depth, depth_vis_path)
    else:
        depth_path = None
        depth_vis_path = None

    if opacity is not None:
        save_float_array(opacity, opacity_path)
    else:
        opacity_path = None

    touched_count = None
    if n_touched is not None:
        if torch.is_tensor(n_touched):
            touched_count = int((n_touched > 0).sum().item())
        else:
            touched_count = int(n_touched)

    meta = {
        "prefix": prefix,
        "prev_kf_idx": None if prev_kf_idx is None else int(prev_kf_idx),
        "curr_kf_idx": None if curr_kf_idx is None else int(curr_kf_idx),
        "source_frame_idx": None if source_frame_idx is None else int(source_frame_idx),
        "alpha": float(alpha),
        "pose_mode": pose_mode,
        "image_path": image_path,
        "depth_path": depth_path,
        "depth_vis_path": depth_vis_path,
        "opacity_path": opacity_path,
        "valid_ratio": valid_ratio,
        "touched_count": touched_count,
        "R_w2c": _to_numpy(pseudo_cam.R).astype(float).tolist(),
        "T_w2c": _to_numpy(pseudo_cam.T).reshape(3).astype(float).tolist(),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "pair": (prev_kf_idx, curr_kf_idx),
        "alpha": alpha,
        "pose_mode": pose_mode,
        "source_frame_idx": source_frame_idx,
        "pseudo_cam": pseudo_cam,
        "image_path": image_path,
        "depth_path": depth_path,
        "depth_vis_path": depth_vis_path,
        "opacity_path": opacity_path,
        "meta_path": meta_path,
        "rgb": rgb,
        "depth": depth.detach().cpu() if depth is not None else None,
        "opacity": opacity.detach().cpu() if opacity is not None else None,
        "valid_ratio": valid_ratio,
        "touched_count": touched_count,
        "render_pkg": render_pkg,
    }