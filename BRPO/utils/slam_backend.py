import random
import time
import copy

import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
import os

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from utils.init_pose import save_depth_comparison

import torch.nn.functional as F
from PIL import Image
from gaussian_splatting.utils.general_utils import inverse_sigmoid


class BackEnd(mp.Process):
    def __init__(self, config, save_dir=None):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False
        self.save_dir = save_dir

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.theta = 0

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.global_BA_itr_num = self.config["Training"]["global_BA_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
    # Insert new Gaussians into the Gaussian scene based on the new keyframe's viewpoint and geometry
    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )
        
    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()
    # Initialize the SLAM map by optimizing Gaussians through multiple iterations
    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, viewpoint, depth=depth,initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(  
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(                 
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:  
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()                         
                self.gaussians.optimizer.zero_grad(set_to_none=True)    

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()   
        Log("Initialized map")
        return render_pkg
    # Optimize keyframe poses and Gaussians scene
    def map(self, current_window, prune=False, iters=1, up_pose = True):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)            
        for cam_idx, viewpoint in self.viewpoints.items():  # Add viewpoints outside the current window to the random_viewpoint_stack
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)        
            
        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []                 
            visibility_filter_acm = []                      
            radii_acm = []                                  
            n_touched_acm = []                            

            keyframes_opt = []          

            for cam_idx in range(len(current_window)):      # For each keyframe in the current window, perform rendering and compute loss
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (                                          
                    image,
                    viewspace_point_tensor,                 
                    visibility_filter,                     
                    radii,                                  
                    depth,                                 
                    opacity,                                
                    n_touched,                              
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)     
                
            # In each iteration, randomly select two non-window keyframes for optimization
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:     
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                
            # isotropic regularization
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}            
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # Only prune on the last iteration and when we have full window
                if prune:     
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = self.config["Training"]["prune_num"]  # prune parameter
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:       
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (                
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian) :
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                if up_pose:
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]
                        if viewpoint.uid == 0:
                            continue
                        update_pose(viewpoint)
        return gaussian_split
                
    def load_rgb_tensor(self, image_path, device="cuda"):
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        return tensor

    def load_mask_tensor(self, mask_path, device="cuda"):
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask).astype(np.float32) / 255.0
        tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
        return tensor
    
    def compute_depth_scale_consistency_loss(
        self,
        pred_depth,
        target_depth,
        mask,
        eps=1e-6,
    ):
        valid = (
            (mask > 0)
            & torch.isfinite(pred_depth)
            & torch.isfinite(target_depth)
            & (pred_depth > eps)
            & (target_depth > eps)
        )

        if valid.sum() < 1:
            return torch.zeros((), device=pred_depth.device, dtype=pred_depth.dtype)

        pred_valid = pred_depth[valid]
        target_valid = target_depth[valid]

        scale = torch.median(target_valid / (pred_valid + eps)).detach()
        pred_scaled = pred_valid * scale

        return torch.mean(torch.abs(pred_scaled - target_valid))

    def compute_masked_rgbd_loss(
        self,
        image,
        target,
        pred_depth,
        target_depth,
        mask,
        target_depth_mask=None,
    ):
        eps = 1e-6

        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        if mask.shape[-2:] != image.shape[-2:]:
            mask = torch.nn.functional.interpolate(
                mask,
                size=image.shape[-2:],
                mode="nearest",
            )

        rgb_mask = mask

        rgb_diff = torch.abs(image - target)
        rgb_weight_sum = rgb_mask.sum().clamp_min(eps)
        l_rgb = (rgb_diff * rgb_mask).sum() / rgb_weight_sum

        l_depth = torch.zeros_like(l_rgb)
        l_scale = torch.zeros_like(l_rgb)

        if pred_depth is not None and target_depth is not None:
            depth_t = target_depth

            if not torch.is_tensor(depth_t):
                depth_t = torch.tensor(depth_t, device=image.device, dtype=torch.float32)
            else:
                depth_t = depth_t.to(image.device).float()

            if depth_t.dim() == 2:
                depth_t = depth_t.unsqueeze(0).unsqueeze(0)
            elif depth_t.dim() == 3:
                depth_t = depth_t.unsqueeze(0)

            if pred_depth.dim() == 2:
                pred_depth = pred_depth.unsqueeze(0).unsqueeze(0)
            elif pred_depth.dim() == 3:
                pred_depth = pred_depth.unsqueeze(0)

            if depth_t.shape[-2:] != pred_depth.shape[-2:]:
                depth_t = torch.nn.functional.interpolate(
                    depth_t,
                    size=pred_depth.shape[-2:],
                    mode="nearest",
                )

            depth_mask = mask
            if depth_mask.shape[-2:] != pred_depth.shape[-2:]:
                depth_mask = torch.nn.functional.interpolate(
                    depth_mask,
                    size=pred_depth.shape[-2:],
                    mode="nearest",
                )

            valid_depth = ((depth_t > 0) & torch.isfinite(depth_t)).float()

            if target_depth_mask is not None:
                if not torch.is_tensor(target_depth_mask):
                    target_depth_mask = torch.tensor(
                        target_depth_mask,
                        device=image.device,
                        dtype=torch.float32,
                    )
                else:
                    target_depth_mask = target_depth_mask.to(image.device).float()

                if target_depth_mask.dim() == 2:
                    target_depth_mask = target_depth_mask.unsqueeze(0).unsqueeze(0)
                elif target_depth_mask.dim() == 3:
                    target_depth_mask = target_depth_mask.unsqueeze(0)

                if target_depth_mask.shape[-2:] != pred_depth.shape[-2:]:
                    target_depth_mask = torch.nn.functional.interpolate(
                        target_depth_mask,
                        size=pred_depth.shape[-2:],
                        mode="nearest",
                    )

                valid_depth = valid_depth * (target_depth_mask > 0).float()

            depth_mask = depth_mask * valid_depth

            depth_weight_sum = depth_mask.sum().clamp_min(eps)
            l_depth = (torch.abs(pred_depth - depth_t) * depth_mask).sum() / depth_weight_sum

            l_scale = self.compute_depth_scale_consistency_loss(
                pred_depth=pred_depth,
                target_depth=depth_t,
                mask=depth_mask,
            )

        beta = self.config["Training"].get("pseudo_rgbd_beta", 0.98)
        lambda_scale = self.config["Training"].get("lambda_pseudo_scale", 0.001)

        loss = beta * l_rgb + (1.0 - beta) * l_depth + lambda_scale * l_scale

        mask_sum = float(mask.sum().detach().cpu())
        valid_ratio = mask_sum / float(mask.numel())

        depth_mask_sum = 0.0
        depth_valid_ratio = 0.0
        if "depth_mask" in locals():
            depth_mask_sum = float(depth_mask.sum().detach().cpu())
            depth_valid_ratio = depth_mask_sum / float(depth_mask.numel())

        info = {
            "l_rgb": float(l_rgb.detach().cpu()),
            "l_depth": float(l_depth.detach().cpu()),
            "l_scale": float(l_scale.detach().cpu()),
            "mask_sum": float(mask.sum().detach().cpu()),
            "valid_ratio": valid_ratio,
            "depth_mask_sum": depth_mask_sum,
            "depth_valid_ratio": depth_valid_ratio,
        }

        return loss, info
    
    def project_gaussians_to_pseudo_camera(self, pseudo_cam):
        xyz = self.gaussians.get_xyz
        device = xyz.device
        N = xyz.shape[0]

        ones = torch.ones((N, 1), device=device, dtype=xyz.dtype)
        xyz_h = torch.cat([xyz, ones], dim=1)

        view = xyz_h @ pseudo_cam.world_view_transform
        z_cam = view[:, 2]

        clip = xyz_h @ pseudo_cam.full_proj_transform
        ndc = clip[:, :3] / (clip[:, 3:4] + 1e-8)

        W = int(pseudo_cam.image_width)
        H = int(pseudo_cam.image_height)

        px = (ndc[:, 0] + 1.0) * 0.5 * (W - 1)
        py = (1.0 - ndc[:, 1]) * 0.5 * (H - 1)

        in_img = (
            (px >= 0)
            & (px <= W - 1)
            & (py >= 0)
            & (py <= H - 1)
            & torch.isfinite(z_cam)
            & (z_cam > 0)
        )

        return px, py, z_cam, in_img


    def sample_map_at_gaussian_centers(self, map_1ch, px, py):
        """
        map_1ch: [1, 1, H, W]
        return: [N]
        """
        H, W = map_1ch.shape[-2:]

        gx = 2.0 * px / max(W - 1, 1) - 1.0
        gy = 2.0 * py / max(H - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)

        sampled = F.grid_sample(
            map_1ch,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )

        return sampled.view(-1)

    def pseudo_refine(
        self,
        pair,
        pseudo_cam,
        fused_path,
        mask_path,
        alpha,
        target_depth=None,
        target_depth_mask=None,
        iters=None,
        lambda_pseudo=None,
    ):
        """
        BRPO-style backend optimization.

        Stage 1:
            stabilize pseudo pose and exposure while keeping Gaussians fixed.

        Stage 2:
            jointly optimize pseudo pose, exposure, and Gaussians with
            confidence-masked RGB-D loss.
        """
        prev_kf, curr_kf = pair

        if lambda_pseudo is None:
            lambda_pseudo = self.config["Training"].get("lambda_pseudo", 0.0025)

        lambda_eff = self.config["Training"].get("lambda_pseudo", 0.0005)

        stabilize_iters = self.config["Training"].get("pseudo_pose_stabilize_iters", 3)
        joint_iters = self.config["Training"].get("pseudo_joint_iters", 5)

        if iters is not None:
            joint_iters = iters

        target = self.load_rgb_tensor(fused_path, device=self.device)
        mask = self.load_mask_tensor(mask_path, device=self.device)

        optimize_pseudo_pose = self.config["Training"].get("optimize_pseudo_pose", True)
        optimize_pseudo_exposure = self.config["Training"].get("optimize_pseudo_exposure", True)

        pseudo_aux_optimizer = None
        opt_params = []

        if optimize_pseudo_pose:
            pseudo_cam.cam_rot_delta.data.zero_()
            pseudo_cam.cam_trans_delta.data.zero_()

            opt_params.append({
                "params": [pseudo_cam.cam_rot_delta],
                "lr": self.config["Training"].get("pseudo_pose_lr_rot", 1e-4),
                "name": f"pseudo_rot_{pseudo_cam.uid}",
            })
            opt_params.append({
                "params": [pseudo_cam.cam_trans_delta],
                "lr": self.config["Training"].get("pseudo_pose_lr_trans", 1e-4),
                "name": f"pseudo_trans_{pseudo_cam.uid}",
            })

        if optimize_pseudo_exposure:
            pseudo_cam.exposure_a.data.zero_()
            pseudo_cam.exposure_b.data.zero_()

            opt_params.append({
                "params": [pseudo_cam.exposure_a],
                "lr": self.config["Training"].get("pseudo_exposure_lr_a", 1e-4),
                "name": f"pseudo_exposure_a_{pseudo_cam.uid}",
            })
            opt_params.append({
                "params": [pseudo_cam.exposure_b],
                "lr": self.config["Training"].get("pseudo_exposure_lr_b", 1e-4),
                "name": f"pseudo_exposure_b_{pseudo_cam.uid}",
            })

        if len(opt_params) > 0:
            pseudo_aux_optimizer = torch.optim.Adam(opt_params)

        Log(
            f"[PseudoJointOpt] start {prev_kf}->{curr_kf}, "
            f"alpha={alpha:.2f}, stabilize_iters={stabilize_iters}, "

            f"opt_pose={optimize_pseudo_pose}, opt_exposure={optimize_pseudo_exposure}"
        )

        total_iters = stabilize_iters + joint_iters

        for refine_iter in range(total_iters):
            self.iteration_count += 1

            is_stabilize_stage = refine_iter < stabilize_iters
            stage_name = "pose_stabilize" if is_stabilize_stage else "joint"

            if pseudo_aux_optimizer is not None:
                pseudo_aux_optimizer.zero_grad(set_to_none=True)

            self.gaussians.optimizer.zero_grad(set_to_none=True)

            render_pkg = render(
                pseudo_cam,
                self.gaussians,
                self.pipeline_params,
                self.background,
            )

            image = render_pkg["render"].unsqueeze(0)
            pred_depth = render_pkg["depth"]

            if optimize_pseudo_exposure:
                image = torch.exp(pseudo_cam.exposure_a) * image + pseudo_cam.exposure_b
                image = torch.clamp(image, 0.0, 1.0)
            
            mask_for_loss = mask

            if target_depth_mask is not None:
                if torch.is_tensor(target_depth_mask):
                    depth_mask_t = target_depth_mask.detach().to(self.device).float()
                else:
                    depth_mask_t = torch.tensor(
                        target_depth_mask,
                        device=self.device,
                        dtype=torch.float32,
                    )

                if depth_mask_t.dim() == 2:
                    depth_mask_t = depth_mask_t.unsqueeze(0).unsqueeze(0)
                elif depth_mask_t.dim() == 3:
                    depth_mask_t = depth_mask_t.unsqueeze(0)

                if mask_for_loss.dim() == 2:
                    mask_for_loss = mask_for_loss.unsqueeze(0).unsqueeze(0)
                elif mask_for_loss.dim() == 3:
                    mask_for_loss = mask_for_loss.unsqueeze(0)

                if depth_mask_t.shape[-2:] != mask_for_loss.shape[-2:]:
                    depth_mask_t = F.interpolate(
                        depth_mask_t,
                        size=mask_for_loss.shape[-2:],
                        mode="nearest",
                    )

                mask_for_loss = mask_for_loss * depth_mask_t

            loss_core, stats = self.compute_masked_rgbd_loss(
                image=image,
                target=target,
                pred_depth=pred_depth,
                target_depth=target_depth,
                mask=mask_for_loss,
                target_depth_mask=target_depth_mask,
            )

            loss = lambda_eff * loss_core
            loss.backward()

            if pseudo_aux_optimizer is not None:
                pseudo_aux_optimizer.step()

            if optimize_pseudo_pose:
                update_pose(pseudo_cam)

            if not is_stabilize_stage:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration_count)

            self.gaussians.optimizer.zero_grad(set_to_none=True)
            if pseudo_aux_optimizer is not None:
                pseudo_aux_optimizer.zero_grad(set_to_none=True)

            if refine_iter == 0 or refine_iter == total_iters - 1:
                Log(
                    f"[PseudoJointOpt] {prev_kf}->{curr_kf} "
                    f"iter={refine_iter + 1}/{total_iters}, "
                    f"stage={stage_name}, "
                    f"loss={loss.item():.6f}, "
                    f"rgb={stats['l_rgb']:.6f}, "
                    f"depth={stats['l_depth']:.6f}, "
                    f"scale={stats['l_scale']:.6f}, "
                    f"valid={stats['valid_ratio']:.3f}"
                )

    # Run color refinement as a post-processing step after SLAM
    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())      
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]      
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():       
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(26000)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
            
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)
    # Main execution loop: 
    # process backend messages, perform initialization, optimize keyframe map, color refinement,
    # synchronize data, and push updates to the frontend
    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:       
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()

                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    self.theta = data[5]
                    theta_value = self.theta.item()
                    print("current keyframe ",cur_frame_idx,'window is ',current_window)

                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_nosingle = self.config["Training"]["mapping_itr_nosingle"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else iter_nosingle
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):     
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:        
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf, up_pose=True)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")

                elif data[0] == "pseudo_refine":
                    payload = data[1]

                    pair = payload["pair"]
                    prev_kf, curr_kf = pair

                    template_kf = payload.get("template_kf", prev_kf)
                    if template_kf not in self.viewpoints:
                        Log(
                            f"[PseudoRefine][WARN] template_kf {template_kf} "
                            f"not in backend viewpoints, skip {prev_kf}->{curr_kf}"
                        )
                        continue

                    pseudo_cam = copy.deepcopy(self.viewpoints[template_kf])

                    R = torch.tensor(
                        payload["pseudo_R"],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    T = torch.tensor(
                        payload["pseudo_T"],
                        device=self.device,
                        dtype=torch.float32,
                    )

                    pseudo_cam.uid = int(payload.get("pseudo_uid", 1000000 + curr_kf))
                    pseudo_cam.update_RT(R, T)

                    pseudo_cam.original_image = None
                    pseudo_cam.depth = None
                    pseudo_cam.mono_depth = None
                    pseudo_cam.grad_mask = None

                    target_depth = None
                    target_depth_path = payload.get("target_depth_path", None)
                    if target_depth_path is not None and os.path.exists(target_depth_path):
                        target_depth = np.load(target_depth_path).astype(np.float32)
                    target_depth_mask = None
                    target_depth_mask_path = payload.get("target_depth_mask_path", None)
                    if target_depth_mask_path is not None and os.path.exists(target_depth_mask_path):
                        target_depth_mask = np.load(target_depth_mask_path).astype(np.float32)

                    self.pseudo_refine(
                        pair=pair,
                        pseudo_cam=pseudo_cam,
                        fused_path=payload["fused_path"],
                        mask_path=payload["mask_path"],
                        alpha=payload["alpha"],
                        target_depth=target_depth,
                        target_depth_mask=target_depth_mask,
                    )

                    self.frontend_queue.put(["pseudo_refine_done"])
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
