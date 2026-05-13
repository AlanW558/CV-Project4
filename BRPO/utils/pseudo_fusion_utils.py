import os
import shutil
import numpy as np
import cv2
from PIL import Image

def load_image_float(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def save_image_float(img, path):
    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255.0).astype(np.uint8)
    Image.fromarray(img_u8).save(path)
    return path

def resize_map_to_image(score, h, w):
    if score is None:
        return np.zeros((h, w), dtype=np.float32)
    score = np.asarray(score, dtype=np.float32)
    if score.shape != (h, w):
        score = cv2.resize(score, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(score, 0.0, 1.0)

def overlap_score_residual_fusion(
    base_path,
    prev_path,
    curr_path,
    score_prev,
    score_curr,
    output_path,
    eps=1e-6,
):
    """
    BRPO-style residual fusion.

    I_fused = I_base + W_prev * (I_prev - I_base)
                       + W_curr * (I_curr - I_base)

    This function does NOT compute reprojection scores.
    score_prev and score_curr must be provided by compute_reprojection_overlap_score().
    """
    base = load_image_float(base_path)
    prev = load_image_float(prev_path)
    curr = load_image_float(curr_path)

    h, w = base.shape[:2]

    if prev.shape[:2] != (h, w):
        prev = cv2.resize(prev, (w, h), interpolation=cv2.INTER_LINEAR)
    if curr.shape[:2] != (h, w):
        curr = cv2.resize(curr, (w, h), interpolation=cv2.INTER_LINEAR)

    score_prev = resize_map_to_image(score_prev, h, w)
    score_curr = resize_map_to_image(score_curr, h, w)

    denom = score_prev + score_curr + eps
    w_prev = score_prev / denom
    w_curr = score_curr / denom

    r_prev = prev - base
    r_curr = curr - base

    fused = base + w_prev[..., None] * r_prev + w_curr[..., None] * r_curr
    fused = np.clip(fused, 0.0, 1.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image_float(fused, output_path)

    # Optional visualization.
    cv2.imwrite(
        output_path.replace(".png", "_w_prev.png"),
        (np.clip(w_prev, 0.0, 1.0) * 255).astype(np.uint8),
    )
    cv2.imwrite(
        output_path.replace(".png", "_w_curr.png"),
        (np.clip(w_curr, 0.0, 1.0) * 255).astype(np.uint8),
    )

    return output_path

def align_pseudo_to_reference_lowfreq(
    pseudo_path,
    ref_path,
    mask,
    output_path,
    sigma=15,
    strength=0.5,
    mask_threshold=0.5,
    min_pixels=200,
):
    """
    Low-frequency color alignment.

    The reference real image is only used to estimate low-frequency
    color / illumination correction. It is NOT used as a backend
    optimization target.

    I_aligned = I_pseudo + strength * soft_mask * (Blur(I_ref) - Blur(I_pseudo))

    High-frequency texture remains from pseudo-view.
    """
    pseudo = load_image_float(pseudo_path)
    ref = load_image_float(ref_path)

    h, w = pseudo.shape[:2]

    if ref.shape[:2] != (h, w):
        ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_LINEAR)

    if mask is None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_image_float(pseudo, output_path)
        return output_path, {
            "used": False,
            "reason": "missing_mask",
            "valid_pixels": 0,
        }

    mask = np.asarray(mask, dtype=np.float32)

    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = np.clip(mask, 0.0, 1.0)

    valid = mask >= float(mask_threshold)

    if int(valid.sum()) < int(min_pixels):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_image_float(pseudo, output_path)
        return output_path, {
            "used": False,
            "reason": "not_enough_valid_pixels",
            "valid_pixels": int(valid.sum()),
        }

    low_pseudo = cv2.GaussianBlur(
        pseudo,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )

    low_ref = cv2.GaussianBlur(
        ref,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )

    delta_low = low_ref - low_pseudo

    # Smooth confidence mask to avoid visible seams.
    soft_mask = cv2.GaussianBlur(
        mask,
        ksize=(0, 0),
        sigmaX=max(float(sigma) * 0.5, 1.0),
        sigmaY=max(float(sigma) * 0.5, 1.0),
    )
    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    aligned = pseudo + float(strength) * soft_mask[..., None] * delta_low
    aligned = np.clip(aligned, 0.0, 1.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image_float(aligned, output_path)

    return output_path, {
        "used": True,
        "valid_pixels": int(valid.sum()),
        "sigma": float(sigma),
        "strength": float(strength),
        "mask_threshold": float(mask_threshold),
    }