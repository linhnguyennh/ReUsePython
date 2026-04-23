import os
import numpy as np
import cv2
import trimesh
import logging
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, draw_posed_3d_box, draw_xyz_axis
import nvdiffrast.torch as dr
from server_ws import StreamServerWebSocket
from queue import Empty

class FoundationPoseEstimator:
    def __init__(
        self,
        mesh_file: str,
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        debug: int = 0,
        debug_dir: str = "/tmp/foundationpose_debug",  # always provide a path
    ):
        self.est_refine_iter   = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self._initialized      = False

        self.mesh      = trimesh.load(mesh_file)
        self.mesh.apply_scale(0.001)
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.to_origin = to_origin
        self.bbox      = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

        os.makedirs(debug_dir, exist_ok=True)  # ensure it exists before passing in

        scorer  = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx   = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=scorer,
            refiner=refiner,
            debug=debug,
            debug_dir=debug_dir,  # never None
            glctx=glctx,
        )

        self.prev_pose = None
        self.smooth_alpha = 0.7

        logging.info("FoundationPose initialized")

    def register(self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        First-frame registration. Call this once when you have a mask.
        mask: bool array (H, W)
        K:    camera intrinsics (3, 3)
        Returns 4x4 pose matrix.
        """
        depth_m = depth.astype(np.float32) / 1000.0
        depth_m[mask == 0] = 0
        pose = self.est.register(
            K=K,
            rgb=rgb,
            depth=depth_m,
            ob_mask=mask.astype(bool),
            iteration=self.est_refine_iter,
        )
        self._initialized = True
        self.prev_pose = pose
        return pose

    def track(self, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Subsequent frames — tracking only, no mask needed.
        Returns 4x4 pose matrix.
        """
        depth_m = depth.astype(np.float32) / 1000.0  # uint16 mm → float32 meters
        if not self._initialized:
            raise RuntimeError("Call register() with a mask on the first frame before tracking.")
        
        pose = self.est.track_one(
        rgb=rgb,
        depth=depth_m,
        K=K,
        iteration=self.track_refine_iter,
        )

        if pose is None or np.isnan(pose).any():
            return self.prev_pose  # or skip update

        pose = self.smooth_pose(pose)

        self.prev_pose = pose
        return pose

    def visualize(self, rgb: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Draws bounding box and XYZ axes onto the image. Returns BGR image."""
        center_pose = pose @ np.linalg.inv(self.to_origin)
        vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=self.bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=False)
        return vis  # RGB → BGR for cv2
    
    def smooth_pose(self, pose):
        if self.prev_pose is None:
            return pose
        pose = pose.copy()
        alpha = self.smooth_alpha

        # translation
        pose[:3, 3] = (
            alpha * self.prev_pose[:3, 3] +
            (1 - alpha) * pose[:3, 3]
        )

        # rotation (simple linear blend, not perfect but stable)
        R_new = pose[:3, :3]
        R_old = self.prev_pose[:3, :3]

        R = alpha * R_old + (1 - alpha) * R_new
        U, _, Vt = np.linalg.svd(R)
        pose[:3, :3] = U @ Vt

        return pose