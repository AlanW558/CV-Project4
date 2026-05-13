import random
import time

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


    def compute_spgm_clusters(self):
        """
        Cluster-aware Gaussian grouping.

        BRPO uses cluster-aware drop p_i = r * w_cluster(i) * S_i.
        Since the paper does not specify clustering details, we use voxel clustering
        over Gaussian centers. Dense clusters get larger weights; sparse clusters
        are protected.
        """
        xyz = self.gaussians.get_xyz.detach()
        device = xyz.device
        N = xyz.shape[0]

        if N == 0:
            return (
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, device=device),
            )

        voxel_size = self.config["Training"].get("spgm_voxel_size", 0.20)

        xyz_min = xyz.min(dim=0, keepdim=True).values
        coords = torch.floor((xyz - xyz_min) / voxel_size).long()

        keys = (
            coords[:, 0] * 73856093
            ^ coords[:, 1] * 19349663
            ^ coords[:, 2] * 83492791
        )

        _, cluster_ids, counts = torch.unique(
            keys, return_inverse=True, return_counts=True
        )

        counts_f = counts.float()
        median_count = torch.median(counts_f).clamp(min=1.0)
        density_ratio = counts_f / median_count

        density_unrel = torch.clamp(
            (density_ratio - 1.0)
            / self.config["Training"].get("spgm_density_ratio_clip", 5.0),
            0.0,
            1.0,
        )

        w_cluster = 0.5 + 0.5 * density_unrel
        w_cluster = torch.clamp(w_cluster, 0.5, 1.0)

        return cluster_ids, density_unrel, w_cluster


    def compute_spgm_scores(
        self,
        pseudo_cam,
        target_depth,
        pred_depth,
        image,
        target_resized,
        visibility_filter,
        mask_resized,
    ):
        """
        Compute per-Gaussian unreliability score S_i.

        Online SLAM-safe design:
        - only visible Gaussians
        - only projected low-confidence regions
        - combines RGB residual, depth residual, cluster density, and opacity
        """
        xyz = self.gaussians.get_xyz
        device = xyz.device
        N = xyz.shape[0]

        S = torch.zeros(N, device=device)
        if N == 0:
            return (
                S,
                torch.zeros(N, dtype=torch.long, device=device),
                torch.zeros(0, device=device),
                torch.zeros(N, device=device),
            )

        px, py, _, in_img = self.project_gaussians_to_pseudo_camera(pseudo_cam)
        visible = visibility_filter.bool() & in_img

        cluster_ids, cluster_density_unrel, w_cluster = self.compute_spgm_clusters()

        if mask_resized is not None:
            sampled_conf = self.sample_map_at_gaussian_centers(mask_resized, px, py)
            low_conf = 1.0 - torch.clamp(sampled_conf, 0.0, 1.0)
        else:
            low_conf = torch.zeros(N, device=device)

        rgb_residual_map = torch.abs(image - target_resized).mean(dim=1, keepdim=True)
        sampled_rgb_res = self.sample_map_at_gaussian_centers(rgb_residual_map, px, py)

        rgb_sigma = self.config["Training"].get("spgm_rgb_sigma", 0.10)
        rgb_unrel = 1.0 - torch.exp(-sampled_rgb_res / rgb_sigma)
        rgb_unrel = torch.clamp(rgb_unrel, 0.0, 1.0)

        depth_unrel = torch.zeros(N, device=device)

        if target_depth is not None and pred_depth is not None:
            if hasattr(target_depth, "detach"):
                depth_t = target_depth.detach().to(device).float()
            else:
                depth_t = torch.tensor(target_depth, device=device).float()

            if depth_t.dim() == 2:
                depth_t = depth_t.unsqueeze(0).unsqueeze(0)
            elif depth_t.dim() == 3:
                depth_t = depth_t.unsqueeze(0)

            if pred_depth.dim() == 2:
                pred_d = pred_depth.unsqueeze(0).unsqueeze(0)
            elif pred_depth.dim() == 3:
                pred_d = pred_depth.unsqueeze(0)
            else:
                pred_d = pred_depth

            if depth_t.shape[-2:] != pred_d.shape[-2:]:
                depth_t = F.interpolate(depth_t, size=pred_d.shape[-2:], mode="nearest")

            depth_res_map = torch.abs(pred_d - depth_t) / (depth_t.abs() + 1e-6)
            valid_depth_map = ((depth_t > 0) & torch.isfinite(depth_t)).float()

            sampled_depth_res = self.sample_map_at_gaussian_centers(depth_res_map, px, py)
            sampled_depth_valid = self.sample_map_at_gaussian_centers(valid_depth_map, px, py)

            depth_sigma = self.config["Training"].get("spgm_depth_sigma", 0.15)
            depth_unrel = 1.0 - torch.exp(-sampled_depth_res / depth_sigma)
            depth_unrel = torch.clamp(depth_unrel, 0.0, 1.0)
            depth_unrel = depth_unrel * (sampled_depth_valid > 0.5).float()

        density_unrel = cluster_density_unrel[cluster_ids]

        opacity = self.gaussians.get_opacity.detach().view(-1)
        opacity_unrel = 1.0 - torch.clamp(opacity, 0.0, 1.0)

        w_rgb = self.config["Training"].get("spgm_rgb_weight", 0.35)
        w_depth = self.config["Training"].get("spgm_depth_weight", 0.35)
        w_density = self.config["Training"].get("spgm_density_weight", 0.20)
        w_opacity = self.config["Training"].get("spgm_opacity_weight", 0.10)

        S = (
            w_rgb * rgb_unrel
            + w_depth * depth_unrel
            + w_density * density_unrel
            + w_opacity * opacity_unrel
        )

        low_conf_power = self.config["Training"].get("spgm_low_conf_power", 1.0)
        S = S * torch.pow(torch.clamp(low_conf, 0.0, 1.0), low_conf_power)

        S = S * visible.float()
        S = torch.clamp(S, 0.0, 1.0)

        return S, cluster_ids, w_cluster, low_conf


    @torch.no_grad()
    def apply_spgm(
        self,
        pseudo_cam,
        target_depth,
        pred_depth,
        image,
        target_resized,
        visibility_filter,
        mask_resized=None,
    ):
        """
        Online-safe BRPO-style SPGM.

        BRPO form:
            p_i_drop = r * w_cluster(i) * S_i
            m_i ~ Bernoulli(1 - p_i_drop)
            alpha_i <- alpha_i * m_i

        Here default is deterministic expected suppression for online stability.
        """
        if not self.config["Training"].get("use_spgm", False):
            return

        if self.gaussians.get_xyz.shape[0] == 0:
            return

        apply_every = self.config["Training"].get("spgm_apply_every", 3)
        if apply_every <= 0:
            return

        if self.iteration_count % apply_every != 0:
            return

        r = self.config["Training"].get("spgm_drop_rate", 0.001)
        max_drop_prob = self.config["Training"].get("spgm_max_drop_prob", 0.03)
        min_opacity_factor = self.config["Training"].get("spgm_min_opacity_factor", 0.80)

        S, cluster_ids, w_cluster, low_conf = self.compute_spgm_scores(
            pseudo_cam=pseudo_cam,
            target_depth=target_depth,
            pred_depth=pred_depth,
            image=image,
            target_resized=target_resized,
            visibility_filter=visibility_filter,
            mask_resized=mask_resized,
        )

        if S.numel() == 0:
            return

        p_drop = r * w_cluster[cluster_ids] * S
        p_drop = torch.clamp(p_drop, 0.0, max_drop_prob)

        visible = visibility_filter.bool()
        p_drop = p_drop * visible.float()

        if p_drop.max().item() <= 0:
            return

        opacity = self.gaussians.get_opacity.detach().view(-1, 1)
        stochastic = self.config["Training"].get("spgm_stochastic", False)

        if stochastic:
            keep = torch.bernoulli(1.0 - p_drop).view(-1, 1)
            opacity_factor = keep + (1.0 - keep) * min_opacity_factor
        else:
            opacity_factor = 1.0 - p_drop.view(-1, 1) * (1.0 - min_opacity_factor)

        new_opacity = torch.clamp(opacity * opacity_factor, 1e-6, 0.999999)
        self.gaussians._opacity.data = inverse_sigmoid(new_opacity)

        num_visible = int(visible.sum().item())
        num_active = int((p_drop[visible] > 0).sum().item()) if num_visible > 0 else 0
        mean_p = float(p_drop[visible].mean().item()) if num_visible > 0 else 0.0
        max_p = float(p_drop[visible].max().item()) if num_visible > 0 else 0.0
        mean_low_conf = float(low_conf[visible].mean().item()) if num_visible > 0 else 0.0

        Log(
            f"[SPGM] visible={num_visible}, active={num_active}, "
            f"mean_p={mean_p:.5f}, max_p={max_p:.5f}, "
            f"low_conf={mean_low_conf:.3f}, stochastic={stochastic}"
        )

    def pseudo_refine(
        self,
        pair,
        pseudo_cam,
        fused_path,
        mask_path,
        alpha,
        target_depth=None,
        iters=None,
        lambda_pseudo=None,
    ):
        """
        Online Version A:
        Use fused pseudo-view as a low-weight masked auxiliary loss.
        The pseudo-view is NOT added to local window, NOT used for tracking,
        and NOT treated as a keyframe.
        """
        if iters is None:
            iters = self.config["Training"].get("pseudo_refine_iters", 10)

        if lambda_pseudo is None:
            lambda_pseudo = self.config["Training"].get("lambda_pseudo", 0.03)

        prev_kf, curr_kf = pair

        Log(
            f"[PseudoRefine] start {prev_kf}->{curr_kf}, "
            f"alpha={alpha:.2f}, iters={iters}, lambda={lambda_pseudo}"
        )

        target = self.load_rgb_tensor(fused_path, device=self.device)
        mask = self.load_mask_tensor(mask_path, device=self.device)

        # Enable local pose optimization for the pseudo camera.
        optimize_pseudo_pose = self.config["Training"].get("optimize_pseudo_pose", True)
        optimize_pseudo_exposure = self.config["Training"].get("optimize_pseudo_exposure", True)

        pseudo_aux_optimizer = None
        opt_params = []

        if optimize_pseudo_pose:
            pseudo_cam.cam_rot_delta.data.zero_()
            pseudo_cam.cam_trans_delta.data.zero_()

            opt_params.append(
                {
                    "params": [pseudo_cam.cam_rot_delta],
                    "lr": self.config["Training"].get("pseudo_pose_lr_rot", 1e-4),
                    "name": f"pseudo_rot_{pseudo_cam.uid}",
                }
            )
            opt_params.append(
                {
                    "params": [pseudo_cam.cam_trans_delta],
                    "lr": self.config["Training"].get("pseudo_pose_lr_trans", 1e-4),
                    "name": f"pseudo_trans_{pseudo_cam.uid}",
                }
            )

        if optimize_pseudo_exposure:
            # BRPO-style exposure correction: I'_t = a_t I_t + b_t.
            # In this codebase exposure_a is initialized as 0, so we use exp(a) as scale.
            pseudo_cam.exposure_a.data.zero_()
            pseudo_cam.exposure_b.data.zero_()

            opt_params.append(
                {
                    "params": [pseudo_cam.exposure_a],
                    "lr": self.config["Training"].get("pseudo_exposure_lr_a", 1e-3),
                    "name": f"pseudo_exposure_a_{pseudo_cam.uid}",
                }
            )
            opt_params.append(
                {
                    "params": [pseudo_cam.exposure_b],
                    "lr": self.config["Training"].get("pseudo_exposure_lr_b", 1e-3),
                    "name": f"pseudo_exposure_b_{pseudo_cam.uid}",
                }
            )

        if len(opt_params) > 0:
            pseudo_aux_optimizer = torch.optim.Adam(opt_params)

        Log(
            f"[PseudoExposure] optimize_pose={optimize_pseudo_pose}, "
            f"optimize_exposure={optimize_pseudo_exposure}"
        )

        for refine_iter in range(iters):
            self.iteration_count += 1

            if pseudo_aux_optimizer is not None:
                pseudo_aux_optimizer.zero_grad(set_to_none=True)

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
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            _, _, H, W = image.shape

            if target.shape[-2:] != (H, W):
                target_resized = F.interpolate(
                    target, size=(H, W), mode="bilinear", align_corners=False
                )
            else:
                target_resized = target

            if mask.shape[-2:] != (H, W):
                mask_resized = F.interpolate(
                    mask, size=(H, W), mode="nearest"
                )
            else:
                mask_resized = mask

            valid_pixels = (mask_resized > 0).float().sum()
            mask_weight_sum = mask_resized.sum()

            if valid_pixels.item() < self.config["Training"].get("pseudo_min_valid_pixels", 50):
                Log(
                    f"[PseudoRefine][WARN] skip {prev_kf}->{curr_kf}: "
                    f"too few valid mask pixels"
                )
                return

            beta = self.config["Training"].get("pseudo_rgbd_beta", 0.95)
            #epsilon = self.config["Training"].get("charbonnier_epsilon", 1e-3)

            # RGB loss: BRPO-style confidence-masked L1.
            rgb_diff = torch.abs(image - target_resized)
            l_rgb = (rgb_diff * mask_resized).sum() / (
                mask_weight_sum * 3.0 + 1e-6
            )

            # Depth loss: BRPO-style confidence-masked L_d.
            l_depth = torch.tensor(0.0, device=self.device)

            if target_depth is not None:
                if hasattr(target_depth, "detach"):
                    depth_t = target_depth.detach().to(self.device).float()
                else:
                    depth_t = torch.tensor(target_depth, device=self.device).float()

                # target depth -> [1, 1, H, W]
                if depth_t.dim() == 2:
                    depth_t = depth_t.unsqueeze(0).unsqueeze(0)
                elif depth_t.dim() == 3:
                    depth_t = depth_t.unsqueeze(0)
                elif depth_t.dim() == 4:
                    pass
                else:
                    raise ValueError(f"Unsupported target_depth shape: {depth_t.shape}")

                # rendered depth -> [1, 1, H, W]
                if pred_depth.dim() == 2:
                    pred_depth = pred_depth.unsqueeze(0).unsqueeze(0)
                elif pred_depth.dim() == 3:
                    pred_depth = pred_depth.unsqueeze(0)
                elif pred_depth.dim() == 4:
                    pass
                else:
                    raise ValueError(f"Unsupported pred_depth shape: {pred_depth.shape}")

                if depth_t.shape[-2:] != pred_depth.shape[-2:]:
                    depth_t = F.interpolate(depth_t, size=pred_depth.shape[-2:], mode="nearest")

                if mask_resized.shape[-2:] != pred_depth.shape[-2:]:
                    depth_mask = F.interpolate(mask_resized, size=pred_depth.shape[-2:], mode="nearest")
                else:
                    depth_mask = mask_resized

                valid_depth = ((depth_t > 0) & torch.isfinite(depth_t)).float()
                depth_mask = depth_mask * valid_depth

                if (depth_mask > 0).float().sum().item() > self.config["Training"].get("pseudo_min_valid_pixels", 50):
                    # Scale-normalized depth difference is safer in monocular SLAM.
                    depth_diff = torch.abs(pred_depth - depth_t) / (depth_t.abs() + 1e-6)
                    l_depth = (depth_diff * depth_mask).sum() / (depth_mask.sum() + 1e-6)
                else:
                    Log(f"[PseudoRefine][WARN] skip depth loss {prev_kf}->{curr_kf}: too few valid depth pixels")

            pseudo_loss = beta * l_rgb + (1.0 - beta) * l_depth

            lambda_pose_reg = self.config["Training"].get("lambda_pseudo_pose_reg", 1e-3)
            lambda_exposure_reg = self.config["Training"].get("lambda_pseudo_exposure_reg", 1e-4)

            pose_reg = torch.tensor(0.0, device=self.device)
            if optimize_pseudo_pose:
                pose_reg = (
                    pseudo_cam.cam_rot_delta.pow(2).sum()
                    + pseudo_cam.cam_trans_delta.pow(2).sum()
                )

            exposure_reg = torch.tensor(0.0, device=self.device)
            if optimize_pseudo_exposure:
                # Keep exposure near identity: exp(a) ≈ 1 and b ≈ 0.
                exposure_reg = (
                    pseudo_cam.exposure_a.pow(2).sum()
                    + pseudo_cam.exposure_b.pow(2).sum()
                )

            lambda_scale = self.config["Training"].get("lambda_pseudo_scale", 0.001)

            scaling = self.gaussians.get_scaling
            scale_reg = torch.mean(torch.sum(scaling * scaling, dim=1))

            loss = (
                lambda_pseudo * pseudo_loss
                + lambda_scale * scale_reg
                + lambda_pose_reg * pose_reg
                + lambda_exposure_reg * exposure_reg
            )

            if refine_iter == 0:
                exposure_scale = torch.exp(pseudo_cam.exposure_a).item() if optimize_pseudo_exposure else 1.0
                exposure_bias = pseudo_cam.exposure_b.item() if optimize_pseudo_exposure else 0.0
                Log(
                    f"[PseudoRefine] l_rgb={l_rgb.item():.6f}, "
                    f"l_depth={l_depth.item():.6f}, beta={beta:.2f}, "
                    f"lambda={lambda_pseudo}, "
                    f"exposure_scale={exposure_scale:.4f}, "
                    f"exposure_bias={exposure_bias:.4f}, "
                    f"scale_reg={scale_reg.item():.6f}, "
                    f"lambda_scale={lambda_scale}"
                )
            loss.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                self.gaussians.add_densification_stats(
                    viewspace_point_tensor,
                    visibility_filter,
                )

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                if self.config["Training"].get("use_spgm", False):
                    self.apply_spgm(
                        pseudo_cam=pseudo_cam,
                        target_depth=target_depth,
                        pred_depth=pred_depth,
                        image=image.detach(),
                        target_resized=target_resized.detach(),
                        visibility_filter=visibility_filter,
                        mask_resized=mask_resized.detach() if mask_resized is not None else None,
                    )

            # 注意：pseudo_aux_optimizer 不要放在 torch.no_grad() 里
            if pseudo_aux_optimizer is not None:
                pseudo_aux_optimizer.step()
                pseudo_aux_optimizer.zero_grad(set_to_none=True)

                if refine_iter == 0:
                    if optimize_pseudo_pose:
                        rot_norm = pseudo_cam.cam_rot_delta.norm().item()
                        trans_norm = pseudo_cam.cam_trans_delta.norm().item()
                    else:
                        rot_norm = 0.0
                        trans_norm = 0.0

                    if optimize_pseudo_exposure:
                        exposure_scale = torch.exp(pseudo_cam.exposure_a).item()
                        exposure_bias = pseudo_cam.exposure_b.item()
                    else:
                        exposure_scale = 1.0
                        exposure_bias = 0.0

                    Log(
                        f"[PseudoAux] first update "
                        f"rot_delta_norm={rot_norm:.6f}, "
                        f"trans_delta_norm={trans_norm:.6f}, "
                        f"exposure_scale={exposure_scale:.4f}, "
                        f"exposure_bias={exposure_bias:.4f}, "
                    )

                if optimize_pseudo_pose:
                    update_pose(pseudo_cam)

        Log(f"[PseudoRefine] done {prev_kf}->{curr_kf}")
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
                    pair = data[1]
                    pseudo_cam = data[2]
                    fused_path = data[3]
                    mask_path = data[4]
                    alpha = data[5]
                    target_depth = data[6] if len(data) > 6 else None

                    self.pseudo_refine(
                        pair=pair,
                        pseudo_cam=pseudo_cam,
                        fused_path=fused_path,
                        mask_path=mask_path,
                        alpha=alpha,
                        target_depth=target_depth,
                    )

                    self.push_to_frontend("pseudo_refine_done")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
