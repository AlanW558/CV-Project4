import time

import numpy as np
import torch
import torch.multiprocessing as mp
import os

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from utils.init_pose import get_pose, get_depth
from utils.depth_utils import process_depth
from utils.pseudo_view_utils import (
    make_pseudo_camera,
    make_pseudo_camera_from_tracked_frame,
    select_tracked_pseudo_frame,
    select_tracked_pseudo_frame_by_alpha,
    render_coarse_pseudo_view,
    save_tensor_image,
    build_pseudo_depth_by_warping,
)
from utils.difix_refine_utils import DifixRefiner
from utils.pseudo_mask_utils import compute_brpo_confidence_mask, compute_reprojection_overlap_score
from utils.pseudo_fusion_utils import overlap_score_residual_fusion, align_pseudo_to_reference_lowfreq

class FrontEnd(mp.Process):
    def __init__(self, config,model, save_dir=None):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None
        self.save_dir = save_dir

        self.initialized = False            
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        
        self.model = model  # MASt3R Model
        self.theta = 0

        self.pseudo_rendered_pairs = set()

        self.use_difix_refine = True
        self.difix_refiner = None
        self.requested_pseudo_refine = 0

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]       
    # Add a new keyframe. Create valid pixel mask using RGB boundary threshold from config, then generate initial depth map
    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        if len(self.kf_indices) > 0:
            last_kf = self.kf_indices[-1]
            viewpoint_last = self.cameras[last_kf]
            R_last = viewpoint_last.R
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        # Compute angular difference with the previous frame (not used)
        R_now = viewpoint.R
        if len(self.kf_indices) > 1:
            R_now = R_now.to(torch.float32)
            R_last = R_last.to(torch.float32)
            R_diff = torch.matmul(R_last.T, R_now)
            trace_R_diff = torch.trace(R_diff)
            theta_rad = torch.acos((trace_R_diff - 1) / 2)
            theta_deg = torch.rad2deg(theta_rad)
            self.theta = theta_deg
        #print("angular difference is:",self.theta)
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]      # Check if sum of RGB channels exceeds threshold; add a new dimension to match expected shape
        if self.monocular:
            if depth is None:
                initial_depth = torch.from_numpy(viewpoint.mono_depth).unsqueeze(0)     # For the first frame, use MASt3R to estimate depth during map initialization
                print("Initial depth map stats for frame", cur_frame_idx, ":",
                    f"Max: {torch.max(initial_depth).item()}",
                    f"Min: {torch.min(initial_depth).item()}",
                    f"Mean: {torch.mean(initial_depth).item()}",
                    f"Median: {torch.median(initial_depth).item()}",
                    f"Std: {torch.std(initial_depth).item()}")
                initial_depth[~valid_rgb.cpu()] = 0
                return initial_depth[0].numpy()
            else:                               # For non-initial keyframes, use rendered depth
                depth = depth.detach().clone()
                opacity = opacity.detach()
                
                initial_depth = depth
                
                # Compute scale factor and adjust rendered depth (Pointmap Replacement)
                render_depth = initial_depth.cpu().numpy()[0]
                initial_depth, scale_factor, error_mask, num_accurate_pixels = process_depth(render_depth, viewpoint.mono_depth, last_depth = viewpoint_last.mono_depth, 
                                                                                             im1 = viewpoint_last.original_image, im2 = viewpoint.original_image, model = self.model,
                                                                                             patch_size = self.config["depth"]["patch_size"], 
                                                                                             mean_threshold = self.config["depth"]["mean_threshold"], std_threshold = self.config["depth"]["std_threshold"],
                                                                                             error_threshold = self.config["depth"]["error_threshold"], final_error_threshold = self.config["depth"]["final_error_threshold"],
                                                                                             min_accurate_pixels_ratio = self.config["depth"]["min_accurate_pixels_ratio"])

                # Correct MASt3R scale
                viewpoint.mono_depth = viewpoint.mono_depth * scale_factor

                pixel_num = viewpoint.image_height * viewpoint.image_width
                #print("Initialization info for frame", cur_frame_idx, ":", 
                #    f"Max: {np.max(initial_depth)}", f"Min: {np.min(initial_depth)}", f"Mean: {np.mean(initial_depth)}",
                #    f"Median: {np.median(initial_depth)}", f"Std: {np.std(initial_depth)}", f"Scale Factor: {scale_factor}", 
                #    f"Accurate Pixel Ratio: {num_accurate_pixels / pixel_num}", f"Accurate Pixel Ratio: {np.sum(error_mask) / pixel_num}")
                
                valid_rgb_np = valid_rgb.cpu().numpy() if isinstance(valid_rgb, torch.Tensor) else valid_rgb
                if initial_depth.shape == valid_rgb_np.shape[1:]:
                    initial_depth[~valid_rgb_np[0]] = 0 
            return initial_depth
        # Keep ground truth depth usage
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)     
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()      # initial_depth is a 4D tensor (1, C, H, W); extract the first channel as (C, H, W)
    
    # Initialize the SLAM system: clear backend queue, reset state, set current frame to ground-truth pose, 
    # add a new keyframe, and push related info into the backend queue
    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        # get mono_depth from MASt3R
        img = viewpoint.original_image
        viewpoint.mono_depth = get_depth(img, img, self.model)
        
        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)      # Request initialization and push related info into the backend queue
        self.reset = False
   
    def tracking(self, cur_frame_idx, viewpoint):    
        ##=====================Pointmap Anchored Pose Estimation(PAPE)=====================
        # The previous frame
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        pose_prev = getWorld2View2(prev.R, prev.T)
        
        # adjacent keyframe
        last_keyframe_idx = self.current_window[0]
        last_kf = self.cameras[last_keyframe_idx]
        pose_last_kf = getWorld2View2(last_kf.R, last_kf.T)
        img1 = last_kf.original_image
        
        # Estimate the relative pose between the current frame and its adjacent keyframe
        img2 = viewpoint.original_image
        rel_pose, render_depth = get_pose(img1=img1, img2=img2, model=self.model, dist_coeffs=self.dataset.dist_coeffs, 
                            viewpoint=last_kf, gaussians=self.gaussians, pipeline_params=self.pipeline_params, background=self.background)
        
        # get mono_depth from MASt3R
        viewpoint.mono_depth = get_depth(img2, img2, self.model)
        
        # Compute current frame's pose estimation
        identity_matrix = torch.eye(4, device=self.device)
        rel_pose = torch.from_numpy(rel_pose).to(self.device).float()
        # If the relative pose is identity (no motion), treat as a failure and use the previous pose
        if torch.allclose(rel_pose, identity_matrix, atol=1e-6):  
            pose_init = rel_pose @ pose_last_kf
            viewpoint.update_RT(prev.R, prev.T)
        else:
            pose_init = rel_pose @ pose_last_kf
            viewpoint.update_RT(pose_init[:3, :3], pose_init[:3, 3])

        # Use previous frame pose (for ablation)
        #viewpoint.update_RT(prev.R, prev.T)
        
        ## ===================================Pose Optimization=================================
        opt_params = []     # Exposure parameters a and b, used to adjust image brightness
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint) 

            if tracking_itr % 10 == 0:              
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break
        
        self.median_depth = get_median_depth(depth, opacity)        # Median of rendered depth, used to determine whether the frame is a keyframe
        return render_pkg
    
    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)  
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)          
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])        # Get transformation matrix from current frame to previous keyframe; extract translation and compute distance
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check     # Small co-visibility or large camera motion
    
    # Add current frame to the window and remove the least important keyframe based on overlap ratio to keep window size within limit
    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if (point_ratio_2 <= cut_off) and (len(window) > self.config["Training"]["window_size"]):        
            #if (point_ratio_2 <= cut_off):
                to_remove.append(kf_idx)
        # Remove the earliest keyframe among those with overlap below the threshold
        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))
        # If the window is still too large, remove the farthest keyframe
        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)
        return window, removed_frame
    
    # Request to add a new keyframe and push related info into the backend queue
    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, self.theta]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def request_pseudo_refinement(self, pseudo_pkg):
        """
        Send pseudo refinement request to backend.

        IMPORTANT:
        multiprocessing queue must only carry paths / numbers / lists.
        Do NOT send pseudo_cam or torch.Tensor directly.
        """
        prev_kf, curr_kf = pseudo_pkg["pair"]
        pseudo_cam = pseudo_pkg["pseudo_cam"]

        target_depth_path = None
        target_depth = pseudo_pkg.get("target_depth", None)
        target_depth_mask_path = None
        target_depth_mask = pseudo_pkg.get("target_depth_mask", None)

        if target_depth is not None:
            target_depth_dir = os.path.join(self.save_dir, "pseudo_depth_target")
            os.makedirs(target_depth_dir, exist_ok=True)

            target_depth_path = os.path.join(
                target_depth_dir,
                f"{prev_kf:05d}_{curr_kf:05d}_a{pseudo_pkg['alpha']:.2f}_target_depth.npy",
            )

            if torch.is_tensor(target_depth):
                target_depth_np = target_depth.detach().cpu().numpy()
            else:
                target_depth_np = np.asarray(target_depth)

            np.save(target_depth_path, target_depth_np.astype(np.float32))
        if target_depth_mask is not None:
            target_depth_dir = os.path.join(self.save_dir, "pseudo_depth_target")
            os.makedirs(target_depth_dir, exist_ok=True)

            target_depth_mask_path = os.path.join(
                target_depth_dir,
                f"{prev_kf:05d}_{curr_kf:05d}_a{pseudo_pkg['alpha']:.2f}_target_depth_mask.npy",
            )

            if torch.is_tensor(target_depth_mask):
                target_depth_mask_np = target_depth_mask.detach().cpu().numpy()
            else:
                target_depth_mask_np = np.asarray(target_depth_mask)

            np.save(target_depth_mask_path, target_depth_mask_np.astype(np.float32))

        payload = {
            "pair": (int(prev_kf), int(curr_kf)),
            "alpha": float(pseudo_pkg["alpha"]),
            "fused_path": str(pseudo_pkg["fused_path"]),
            "mask_path": str(pseudo_pkg["mask_path"]),
            "target_depth_path": target_depth_path,
            "target_depth_mask_path": target_depth_mask_path,

            "template_kf": int(prev_kf),
            "pseudo_uid": int(getattr(pseudo_cam, "uid", 1000000 + curr_kf)),
            "pseudo_R": pseudo_cam.R.detach().cpu().numpy().astype(np.float32).tolist(),
            "pseudo_T": pseudo_cam.T.detach().cpu().numpy().astype(np.float32).reshape(3).tolist(),
        }

        self.backend_queue.put(["pseudo_refine", payload])
        self.requested_pseudo_refine += 1

        Log(
            f"[PseudoRefine] requested backend refinement for "
            f"{prev_kf}->{curr_kf}, payload=path_pose_only"
        )
    
    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)
    
    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True
    # Synchronize data from backend, including Gaussian scene, occlusion-aware visibility, and keyframe info; update keyframe
    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())
    
    def get_pseudo_alphas(self):
        """
        Example:
            n=1 -> [0.5]
            n=3 -> [0.25, 0.5, 0.75]
            n=5 -> [...]
        """
        n = self.config["Training"].get("pseudo_views_per_pair", 1)

        return [
            (i + 1) / float(n + 1)
            for i in range(n)
        ]
    
    def render_pseudo_for_latest_keyframe(self, alpha=0.5):
        """
        BRPO-style coarse pseudo-view rendering for SLAM.

        Priority:
        1. Use an already tracked non-keyframe pose between two keyframes.
        2. Fallback to interpolated keyframe pose if no tracked frame exists.
        """
        if len(self.kf_indices) < 2:
            Log("[PseudoRender] skip: first keyframe")
            return None

        prev_kf_idx = self.kf_indices[-2]
        curr_kf_idx = self.kf_indices[-1]

        if prev_kf_idx not in self.cameras or curr_kf_idx not in self.cameras:
            Log(
                f"[PseudoRender][WARN] missing camera: "
                f"prev={prev_kf_idx}, curr={curr_kf_idx}"
            )
            return None

        output_dir = os.path.join(self.save_dir, "pseudo_rendered_online")
        os.makedirs(output_dir, exist_ok=True)

        try:
            src_idx = select_tracked_pseudo_frame_by_alpha(
                cameras=self.cameras,
                prev_kf_idx=prev_kf_idx,
                curr_kf_idx=curr_kf_idx,
                alpha=alpha,
            )

            if src_idx is not None:
                pseudo_cam = make_pseudo_camera_from_tracked_frame(
                    self.cameras[src_idx],
                    uid=1_000_000 + curr_kf_idx,
                )
                pose_mode = "tracked_non_keyframe"
                prefix = (
                    f"pseudo_{prev_kf_idx:05d}_{curr_kf_idx:05d}"
                    f"_src{src_idx:05d}"
                )
                Log(
                    f"[PseudoRender] use tracked non-keyframe pose: "
                    f"{prev_kf_idx}->{src_idx}->{curr_kf_idx}"
                )
            else:
                pseudo_cam = make_pseudo_camera(
                    self.cameras[prev_kf_idx],
                    self.cameras[curr_kf_idx],
                    alpha=alpha,
                    uid=1_000_000 + curr_kf_idx,
                )
                pose_mode = "interpolated_keyframe"
                prefix = (
                    f"pseudo_{prev_kf_idx:05d}_{curr_kf_idx:05d}"
                    f"_a{alpha:.2f}"
                )
                Log(
                    f"[PseudoRender] no tracked non-keyframe found, "
                    f"fallback to interpolation: {prev_kf_idx}->{curr_kf_idx}, "
                    f"alpha={alpha:.2f}"
                )

            pseudo_pkg = render_coarse_pseudo_view(
                pseudo_cam=pseudo_cam,
                gaussians=self.gaussians,
                pipe=self.pipeline_params,
                background=self.background,
                output_dir=output_dir,
                prefix=prefix,
                prev_kf_idx=prev_kf_idx,
                curr_kf_idx=curr_kf_idx,
                source_frame_idx=src_idx,
                alpha=alpha,
                pose_mode=pose_mode,
                min_valid_ratio=self.config["Training"].get(
                    "pseudo_min_valid_ratio", 0.01
                ),
            )

            if pseudo_pkg is None:
                Log(
                    f"[PseudoRender][WARN] failed or invalid render: "
                    f"{prev_kf_idx}->{curr_kf_idx}"
                )
                return None

            Log(
                f"[PseudoRender] generated {prev_kf_idx}->{curr_kf_idx}, "
                f"pose_mode={pose_mode}, src={src_idx}, "
                f"valid_ratio={pseudo_pkg.get('valid_ratio', None)}, "
                f"touched={pseudo_pkg.get('touched_count', None)}, "
                f"saved={pseudo_pkg['image_path']}"
            )

            return pseudo_pkg

        except Exception as e:
            Log(
                f"[PseudoRender][ERROR] failed {prev_kf_idx}->{curr_kf_idx}: {repr(e)}"
            )
            return None
        
    def refine_latest_pseudo_bidirectional(self, pseudo_pkg):
        if pseudo_pkg is None:
            return None

        if not self.use_difix_refine:
            return None

        prev_kf_idx, curr_kf_idx = pseudo_pkg["pair"]
        coarse_path = pseudo_pkg["image_path"]

        prev_ref_path = self.dataset.color_paths[prev_kf_idx]
        curr_ref_path = self.dataset.color_paths[curr_kf_idx]

        output_dir = os.path.join(self.save_dir, "pseudo_refined_online")
        os.makedirs(output_dir, exist_ok=True)

        prefix = (
            f"pseudo_{prev_kf_idx:05d}_{curr_kf_idx:05d}"
            f"_a{pseudo_pkg['alpha']:.2f}"
        )

        forward_path = os.path.join(output_dir, f"{prefix}_ref_prev.png")
        backward_path = os.path.join(output_dir, f"{prefix}_ref_curr.png")

        try:
            if self.difix_refiner is None:
                Log("[Difix] loading Difix3D pipeline...")
                self.difix_refiner = DifixRefiner(
                    model_dir=self.config["Training"].get(
                        "difix_model_dir",
                        "/data3/yywong558/part3/difix",
                    ),
                    height=self.config["Training"].get("difix_height", 512),
                    width=self.config["Training"].get("difix_width", 512),
                    prompt=self.config["Training"].get(
                        "difix_prompt",
                        "remove degradation",
                    ),
                    timestep=self.config["Training"].get("difix_timestep", 199),
                )
                Log("[Difix] loaded.")

            Log(
                f"[Difix] restoring {prev_kf_idx}->{curr_kf_idx} "
                f"conditioned on prev reference..."
            )
            self.difix_refiner.refine(
                input_image_path=coarse_path,
                ref_image_path=prev_ref_path,
                output_image_path=forward_path,
            )

            Log(
                f"[Difix] restoring {prev_kf_idx}->{curr_kf_idx} "
                f"conditioned on curr reference..."
            )
            self.difix_refiner.refine(
                input_image_path=coarse_path,
                ref_image_path=curr_ref_path,
                output_image_path=backward_path,
            )
            
            pseudo_pkg["deblur_path"] = coarse_path
            pseudo_pkg["refined_prev_path"] = forward_path
            pseudo_pkg["refined_curr_path"] = backward_path

            Log(
                f"[Difix] saved BRPO bidirectional restorations: "
                f"{forward_path}, {backward_path}"
            )

            return pseudo_pkg

        except Exception as e:
            Log(
                f"[Difix][ERROR] failed restoring "
                f"{prev_kf_idx}->{curr_kf_idx}: {repr(e)}"
            )
            return None
        
    def fuse_latest_pseudo_overlap(self, refined_pkg):
        """
        BRPO Eq.3-Eq.8 overlap-score residual fusion.

        This must be done BEFORE confidence mask inference.
        """
        if refined_pkg is None:
            return None

        prev_kf, curr_kf = refined_pkg["pair"]

        prev_cam = self.cameras[prev_kf]
        curr_cam = self.cameras[curr_kf]
        pseudo_cam = refined_pkg.get("pseudo_cam", None)

        if pseudo_cam is None:
            Log(f"[PseudoFusion][WARN] missing pseudo_cam for {prev_kf}->{curr_kf}")
            return None

        pseudo_depth = refined_pkg.get("depth", None)
        prev_depth = getattr(prev_cam, "mono_depth", None)
        curr_depth = getattr(curr_cam, "mono_depth", None)

        fused_dir = os.path.join(self.save_dir, "pseudo_fused")
        os.makedirs(fused_dir, exist_ok=True)

        prefix = (
            f"{prev_kf:05d}_{curr_kf:05d}"
            f"_a{refined_pkg['alpha']:.2f}"
        )
        fused_path = os.path.join(fused_dir, f"{prefix}_fused.png")

        try:
            depth_tol = self.config["Training"].get("pseudo_reproj_depth_tol", 0.30)

            score_prev, score_curr = compute_reprojection_overlap_score(
                pseudo_cam=pseudo_cam,
                prev_cam=prev_cam,
                curr_cam=curr_cam,
                pseudo_depth=pseudo_depth,
                prev_depth=prev_depth,
                curr_depth=curr_depth,
                depth_tol=depth_tol,
            )

            if score_prev is None or score_curr is None:
                Log(f"[PseudoFusion][SKIP] {prev_kf}->{curr_kf}, reason=missing_reprojection_score")
                return None

            result_path = overlap_score_residual_fusion(
                base_path=refined_pkg["image_path"],
                prev_path=refined_pkg["refined_prev_path"],
                curr_path=refined_pkg["refined_curr_path"],
                score_prev=score_prev,
                score_curr=score_curr,
                output_path=fused_path,
            )

            fusion_stats = {
                "overlap_prev_mean": float(np.mean(score_prev)),
                "overlap_curr_mean": float(np.mean(score_curr)),
                "overlap_valid_ratio": float(np.mean((score_prev + score_curr) > 1e-4)),
                "w_prev_mean": float(np.mean(score_prev / (score_prev + score_curr + 1e-6))),
                "w_curr_mean": float(np.mean(score_curr / (score_prev + score_curr + 1e-6))),
            }

            refined_pkg["fused_path"] = result_path
            refined_pkg["score_prev"] = score_prev
            refined_pkg["score_curr"] = score_curr
            refined_pkg["fusion_stats"] = fusion_stats

            Log(
                f"[PseudoFusion] overlap residual fusion {prev_kf}->{curr_kf}, "
                f"valid={fusion_stats['overlap_valid_ratio']:.3f}, "
                f"w_prev={fusion_stats['w_prev_mean']:.3f}, "
                f"w_curr={fusion_stats['w_curr_mean']:.3f}, "
                f"saved={result_path}"
            )

            return refined_pkg

        except Exception as e:
            Log(
                f"[PseudoFusion][ERROR] failed {prev_kf}->{curr_kf}: {repr(e)}"
            )
            return None
        
    def align_latest_pseudo_lowfreq(self, refined_pkg, mask):
        """
        Use the tracked non-keyframe real image only as a low-frequency
        color reference. The backend still optimizes only the aligned
        pseudo-view.

        If the pseudo pose is interpolated and has no source frame, skip.
        """
        if refined_pkg is None:
            return None

        src_idx = refined_pkg.get("source_frame_idx", None)

        if src_idx is None:
            Log("[PseudoAlign] skip low-frequency alignment: no source_frame_idx")
            return refined_pkg

        if not hasattr(self.dataset, "color_paths"):
            Log("[PseudoAlign][WARN] dataset has no color_paths, skip")
            return refined_pkg

        if src_idx < 0 or src_idx >= len(self.dataset.color_paths):
            Log(
                f"[PseudoAlign][WARN] invalid source_frame_idx={src_idx}, "
                f"len(color_paths)={len(self.dataset.color_paths)}"
            )
            return refined_pkg

        ref_path = self.dataset.color_paths[src_idx]

        if ref_path is None or not os.path.exists(ref_path):
            Log(f"[PseudoAlign][WARN] missing real reference image: {ref_path}")
            return refined_pkg

        if "fused_path" not in refined_pkg or not os.path.exists(refined_pkg["fused_path"]):
            Log("[PseudoAlign][WARN] missing fused pseudo image, skip")
            return refined_pkg

        prev_kf, curr_kf = refined_pkg["pair"]

        aligned_dir = os.path.join(self.save_dir, "pseudo_aligned_lowfreq")
        os.makedirs(aligned_dir, exist_ok=True)

        aligned_path = os.path.join(
            aligned_dir,
            f"{prev_kf:05d}_{curr_kf:05d}"
            f"_a{refined_pkg['alpha']:.2f}"
            f"_src{src_idx:05d}_lowfreq.png",
        )

        try:
            aligned_path, align_info = align_pseudo_to_reference_lowfreq(
                pseudo_path=refined_pkg["fused_path"],
                ref_path=ref_path,
                mask=mask,
                output_path=aligned_path,
                sigma=15,
                strength=0.5,
                mask_threshold=0.5,
                min_pixels=200,
            )

            refined_pkg["fused_path_raw"] = refined_pkg["fused_path"]
            refined_pkg["fused_path"] = aligned_path
            refined_pkg["lowfreq_align_info"] = align_info

            Log(
                f"[PseudoAlign] low-frequency aligned with real frame {src_idx}, "
                f"used={align_info.get('used', False)}, "
                f"valid_pixels={align_info.get('valid_pixels', 0)}, "
                f"saved={aligned_path}"
            )

            return refined_pkg

        except Exception as e:
            Log(
                f"[PseudoAlign][WARN] low-frequency alignment failed "
                f"{prev_kf}->{curr_kf}, src={src_idx}: {repr(e)}"
            )
            return refined_pkg
    
    # Clear current frame's camera data; clear CUDA cache every 10 frames
    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()
            
    # Main execution loop: process messages in frontend and backend queues, perform tracking, keyframe management, 
    # synchronize data, clean up resources, and save results
    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(    
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)      
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():      
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():    
                tic.record()
                if cur_frame_idx >= len(self.dataset):  # If current frame index exceeds dataset length, evaluate results, save, and exit the loop
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                
                if self.requested_pseudo_refine > 0:
                    time.sleep(0.01)
                    continue
               
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )
                
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval    # Frame interval is used as a criterion for keyframe selection
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,         
                )
                if len(self.current_window) < self.window_size:    
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:      
                    create_kf = check_time and create_kf
                create_kf = check_time and create_kf
                if create_kf:     
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )       
                    depth_map = self.add_new_keyframe(     
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )

                    #self.render_pseudo_for_latest_keyframe(alpha=0.5)

                    self.request_keyframe(    
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1          

                if (        # Evaluate camera pose if the conditions are satisfied
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()       
                if create_kf:
                    # throttle at 3fps when keyframe is added   
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:      
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)
                    self.requested_pseudo_refine -= 1
                    if self.requested_pseudo_refine == 0:
                        Log("[PseudoRefine] frontend synced pseudo-refined map")

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                    alphas = self.get_pseudo_alphas()

                    for alpha in alphas:

                        pseudo_pkg = self.render_pseudo_for_latest_keyframe(alpha=alpha)

                        if pseudo_pkg is None:
                            continue

                        Log(
                            f"[PseudoRender] ready for Difix refinement: "
                            f"{pseudo_pkg['pair'][0]}->{pseudo_pkg['pair'][1]}, "
                            f"alpha={alpha:.2f}"
                        )
                        refined_pkg = self.refine_latest_pseudo_bidirectional(pseudo_pkg)

                        if refined_pkg is not None:
                            refined_pkg = self.fuse_latest_pseudo_overlap(refined_pkg)

                            if refined_pkg is None:
                                Log("[PseudoFusion][WARN] skip: fusion failed")
                                continue

                            prev_kf, curr_kf = refined_pkg["pair"]

                            mask_dir = os.path.join(self.save_dir, "pseudo_mask")
                            os.makedirs(mask_dir, exist_ok=True)

                            prefix = (
                                f"{prev_kf:05d}_{curr_kf:05d}"
                                f"_a{refined_pkg['alpha']:.2f}"
                            )

                            prev_ref_path = self.dataset.color_paths[prev_kf]
                            curr_ref_path = self.dataset.color_paths[curr_kf]

                            prev_cam = self.cameras[prev_kf]
                            curr_cam = self.cameras[curr_kf]
                            pseudo_cam = refined_pkg.get("pseudo_cam", None)

                            prev_depth = getattr(prev_cam, "mono_depth", None)
                            curr_depth = getattr(curr_cam, "mono_depth", None)

                            # BRPO Eq.9: confidence mask must be inferred on fused image.
                            mask, mask_stats = compute_brpo_confidence_mask(
                                pseudo_path=refined_pkg["fused_path"],
                                prev_ref_path=prev_ref_path,
                                curr_ref_path=curr_ref_path,
                                matcher=self.model,
                                opacity=refined_pkg.get("opacity", None),
                                depth=refined_pkg.get("depth", None),
                                opacity_threshold=self.config["Training"].get("pseudo_opacity_threshold", 0.3),
                                use_depth_valid=True,
                                point_radius=self.config["Training"].get("pseudo_match_radius", 2),
                                size=self.config["Training"].get("pseudo_match_size", 512),
                                subsample=self.config["Training"].get("pseudo_match_subsample", 8),
                                save_dir=mask_dir,
                                prefix=prefix,
                                device=self.device,

                                pseudo_cam=pseudo_cam,
                                prev_cam=prev_cam,
                                curr_cam=curr_cam,
                                prev_depth=prev_depth,
                                curr_depth=curr_depth,
                                use_reprojection=self.config["Training"].get("pseudo_use_reprojection", True),
                                depth_tol=self.config["Training"].get("pseudo_reproj_depth_tol", 0.15),
                                corr_threshold=self.config["Training"].get("pseudo_corr_threshold", 0.15),
                                overlap_threshold=self.config["Training"].get("pseudo_overlap_threshold", 0.05),
                            )

                            mask_path = os.path.join(mask_dir, f"{prefix}_final_mask.png")

                            refined_pkg["mask"] = mask
                            refined_pkg["mask_path"] = mask_path
                            refined_pkg["mask_stats"] = mask_stats
                            if self.config["Training"].get("pseudo_use_lowfreq_align", False):
                                refined_pkg = self.align_latest_pseudo_lowfreq(refined_pkg, mask)
                            else:
                                Log("[PseudoAlign] low-frequency alignment disabled")

                            if refined_pkg is None:
                                Log("[PseudoAlign][WARN] alignment returned None, skip pseudo refinement")
                                continue

                            if refined_pkg is None:
                                Log("[PseudoAlign][WARN] alignment returned None, skip pseudo refinement")
                                continue
                            # ------------------------------------------------------------
                            # Build reprojection-based pseudo target depth.
                            # This replaces image-space interpolation of prev/curr depth.
                            # ------------------------------------------------------------
                            prev_kf, curr_kf = refined_pkg["pair"]
                            prev_cam = self.cameras[prev_kf]
                            curr_cam = self.cameras[curr_kf]
                            pseudo_cam = refined_pkg["pseudo_cam"]

                            prev_depth = getattr(prev_cam, "mono_depth", None)
                            curr_depth = getattr(curr_cam, "mono_depth", None)
                            pseudo_render_depth = refined_pkg.get("depth", None)

                            if prev_depth is not None and curr_depth is not None:
                                try:
                                    target_depth, target_depth_mask = build_pseudo_depth_by_warping(
                                        pseudo_cam=pseudo_cam,
                                        prev_cam=prev_cam,
                                        curr_cam=curr_cam,
                                        prev_depth=prev_depth,
                                        curr_depth=curr_depth,
                                        alpha=refined_pkg.get("alpha", alpha),
                                        pseudo_render_depth=pseudo_render_depth,
                                    )

                                    refined_pkg["target_depth"] = target_depth
                                    refined_pkg["target_depth_mask"] = target_depth_mask

                                    Log(
                                        f"[PseudoDepth] warped target depth built for "
                                        f"{prev_kf}->{curr_kf}, "
                                        f"valid_ratio={(target_depth_mask > 0).mean():.4f}"
                                    )

                                except Exception as e:
                                    refined_pkg["target_depth"] = None
                                    refined_pkg["target_depth_mask"] = None
                                    Log(
                                        f"[PseudoDepth][WARN] failed building warped target depth "
                                        f"{prev_kf}->{curr_kf}: {repr(e)}"
                                    )
                            else:
                                refined_pkg["target_depth"] = None
                                refined_pkg["target_depth_mask"] = None
                                Log(
                                    f"[PseudoDepth][WARN] missing prev/curr depth for "
                                    f"{prev_kf}->{curr_kf}"
                                )

                            valid_ratio = float((mask > 0).mean())
                            weighted_ratio = float(mask.mean())

                            prev_kf, curr_kf = refined_pkg["pair"]

                            pseudo_cam = refined_pkg["pseudo_cam"]
                            prev_cam = self.cameras[prev_kf]
                            curr_cam = self.cameras[curr_kf]

                            prev_depth = getattr(prev_cam, "mono_depth", None)
                            curr_depth = getattr(curr_cam, "mono_depth", None)
                            pseudo_render_depth = refined_pkg.get("depth", None)

                            if prev_depth is not None and curr_depth is not None and pseudo_render_depth is not None:
                                try:
                                    target_depth, target_depth_mask = build_pseudo_depth_by_warping(
                                        pseudo_cam=pseudo_cam,
                                        prev_cam=prev_cam,
                                        curr_cam=curr_cam,
                                        prev_depth=prev_depth,
                                        curr_depth=curr_depth,
                                        alpha=refined_pkg.get("alpha", 0.5),
                                        pseudo_render_depth=pseudo_render_depth,
                                    )

                                    refined_pkg["target_depth"] = target_depth
                                    refined_pkg["target_depth_mask"] = target_depth_mask

                                    Log(
                                        f"[PseudoDepth] warped target depth built for "
                                        f"{prev_kf}->{curr_kf}, "
                                        f"alpha={refined_pkg.get('alpha', 0.5):.2f}, "
                                        f"valid_ratio={float((target_depth_mask > 0).mean()):.4f}"
                                    )

                                except Exception as e:
                                    refined_pkg["target_depth"] = None
                                    refined_pkg["target_depth_mask"] = None

                                    Log(
                                        f"[PseudoDepth][WARN] failed to build warped target depth "
                                        f"{prev_kf}->{curr_kf}, "
                                        f"alpha={refined_pkg.get('alpha', 0.5):.2f}: {repr(e)}"
                                    )
                            else:
                                refined_pkg["target_depth"] = None
                                refined_pkg["target_depth_mask"] = None

                                Log(
                                    f"[PseudoDepth][WARN] missing depth for "
                                    f"{prev_kf}->{curr_kf}, "
                                    f"prev_depth={prev_depth is not None}, "
                                    f"curr_depth={curr_depth is not None}, "
                                    f"pseudo_depth={pseudo_render_depth is not None}"
                                )

                            self.request_pseudo_refinement(refined_pkg)

                            Log(
                                f"[PseudoMask] {prev_kf}->{curr_kf} "
                                f"valid={valid_ratio:.3f}, "
                                f"weighted={weighted_ratio:.3f}, "
                                f"full={mask_stats.get('full_ratio', 0.0):.3f}, "
                                f"half={mask_stats.get('half_ratio', 0.0):.3f}, "
                                f"corr_prev={mask_stats.get('corr_prev_mean', 0.0):.3f}, "
                                f"corr_curr={mask_stats.get('corr_curr_mean', 0.0):.3f}, "
                                f"reproj_prev={mask_stats.get('reproj_prev_mean', 0.0):.3f}, "
                                f"reproj_curr={mask_stats.get('reproj_curr_mean', 0.0):.3f}"
                            )

                            # 存进 pkg（后面做 loss 用）
                            refined_pkg["mask"] = mask

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
