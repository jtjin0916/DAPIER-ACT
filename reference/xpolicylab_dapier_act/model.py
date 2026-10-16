import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


ROBOTWIN_ROOT = Path(__file__).resolve().parents[3]

if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))


from dapier_act.model_multimodal import DapierACTMultimodal
from dapier_act.pointcloud_preprocess import (
    TaskFrameConfig,
    transform_and_crop,
)


def rgb_to_tensor(image):
    """
    Same preprocessing as training:

        uint8 RGB [H,W,3]
        -> float32 [1,3,H,W]
        -> /255

    No resize.
    No ImageNet mean/std normalization.
    """
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(
            f"Expected RGB image [H,W,C], got {image.shape}"
        )

    # Be tolerant of RGBA.
    if image.shape[2] == 4:
        image = image[:, :, :3]

    if image.shape[2] != 3:
        raise ValueError(
            f"Expected 3 RGB channels, got {image.shape}"
        )

    image = np.ascontiguousarray(image)

    tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    return tensor.unsqueeze(0)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = "joint"
        self.env_cfg_type = model_cfg["env_cfg_type"]

        self.robot_action_dim_info = (
            get_robot_action_dim_info(
                self.env_cfg_type
            )
        )

        action_dim = (
            sum(self.robot_action_dim_info["arm_dim"])
            + sum(self.robot_action_dim_info["ee_dim"])
        )

        if action_dim != 12:
            raise ValueError(
                f"Expected action dim 12, got {action_dim}"
            )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ----------------------------------------------------
        # Resolve checkpoint / stats
        # ----------------------------------------------------
        ckpt_setting = Path(
            str(model_cfg["ckpt_name"])
        ).expanduser()

        if not ckpt_setting.is_absolute():
            ckpt_setting = (
                ROBOTWIN_ROOT
                / ckpt_setting
            )

        if ckpt_setting.is_dir():
            checkpoint_path = (
                ckpt_setting
                / "policy_best.pt"
            )
            stats_path = (
                ckpt_setting
                / "dataset_stats.pkl"
            )
        else:
            checkpoint_path = ckpt_setting
            stats_path = (
                checkpoint_path.parent
                / "dataset_stats.pkl"
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                checkpoint_path
            )

        if not stats_path.exists():
            raise FileNotFoundError(
                stats_path
            )

        print(
            "[DAPIER_ACT] checkpoint:",
            checkpoint_path,
        )
        print(
            "[DAPIER_ACT] stats:",
            stats_path,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

        model_type = checkpoint.get(
            "model_type"
        )

        if model_type != "multimodal_dp3head256_resnet18":
            raise ValueError(
                "This adapter expects "
                "multimodal_dp3head256_resnet18, "
                f"got {model_type!r}"
            )

        # ----------------------------------------------------
        # Model configuration
        # ----------------------------------------------------
        self.chunk_size = int(
            checkpoint["chunk_size"]
        )

        self.num_point_samples = int(
            checkpoint.get(
                "num_point_samples",
                512,
            )
        )

        self.candidate_points = int(
            checkpoint.get(
                "candidate_points",
                8192,
            )
        )

        point_min = np.asarray(
            checkpoint.get(
                "point_min",
                self.stats["point_min"],
            ),
            dtype=np.float32,
        )

        point_max = np.asarray(
            checkpoint.get(
                "point_max",
                self.stats["point_max"],
            ),
            dtype=np.float32,
        )

        self.model = DapierACTMultimodal(
            qpos_dim=12,
            action_dim=12,
            chunk_size=self.chunk_size,
            hidden_dim=256,
            latent_dim=32,
            num_point_samples=self.num_point_samples,
            candidate_points=self.candidate_points,
            point_min=point_min,
            point_max=point_max,
            nheads=8,
            encoder_layers=4,
            decoder_layers=4,
            ff_dim=1024,
            dropout=0.0,

            # All trained wrist weights are immediately loaded
            # from checkpoint, so do not download ImageNet
            # weights during deployment.
            pretrained_wrist_backbone=False,
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        self.model.eval()

        # ----------------------------------------------------
        # Task-frame preprocessing.
        #
        # Prefer the exact configuration saved with training.
        # ----------------------------------------------------
        task_cfg = checkpoint.get(
            "task_frame",
            {},
        )

        self.task_frame_cfg = TaskFrameConfig(
            origin_world=tuple(
                task_cfg.get(
                    "origin_world",
                    (0.0, -0.35, 0.74),
                )
            ),
            rpy_deg=tuple(
                task_cfg.get(
                    "rpy_deg",
                    (0.0, 0.0, 90.0),
                )
            ),
            crop_min=tuple(
                task_cfg.get(
                    "crop_min",
                    (0.0, -0.25, 0.005),
                )
            ),
            crop_max=tuple(
                task_cfg.get(
                    "crop_max",
                    (0.40, 0.25, 0.30),
                )
            ),
        )

        # ----------------------------------------------------
        # Train statistics
        # ----------------------------------------------------
        self.qpos_mean = torch.tensor(
            self.stats["qpos_mean"],
            dtype=torch.float32,
            device=self.device,
        )

        self.qpos_std = torch.tensor(
            self.stats["qpos_std"],
            dtype=torch.float32,
            device=self.device,
        )

        self.action_mean = torch.tensor(
            self.stats["action_mean"],
            dtype=torch.float32,
            device=self.device,
        )

        self.action_std = torch.tensor(
            self.stats["action_std"],
            dtype=torch.float32,
            device=self.device,
        )

        # NOTE:
        # Do NOT normalize PointCloud here.
        #
        # DP3PointGlobalEncoder internally performs:
        # candidate reduction -> FPS512 ->
        # point_min/max limits normalization.

        self.qpos = None
        self.pointcloud = None
        self.left_wrist = None
        self.right_wrist = None

        self._debug_printed = False
        self.temporal_step = 0
        self.temporal_history = []

        # ACT temporal aggregation
        self.temporal_k = 0.01
        self.temporal_step = 0
        self.temporal_history = []

        print(
            "[DAPIER_ACT] initialized",
            f"device={self.device}",
            f"chunk={self.chunk_size}",
            f"samples={self.num_point_samples}",
            f"candidates={self.candidate_points}",
        )

        print(
            "[DAPIER_ACT] Task crop:",
            "min=",
            self.task_frame_cfg.crop_min,
            "max=",
            self.task_frame_cfg.crop_max,
        )

        print(
            "[DAPIER_ACT] point limits:",
            "min=",
            point_min.tolist(),
            "max=",
            point_max.tolist(),
        )

    def update_obs(self, obs):
        # ----------------------------------------------------
        # 1. qpos
        # ----------------------------------------------------
        qpos = pack_robot_state(
            obs,
            action_type="joint",
            robot_action_dim_info=(
                self.robot_action_dim_info
            ),
            source_type="obs",
            state_type="state",
        )

        qpos = np.asarray(
            qpos,
            dtype=np.float32,
        )

        if qpos.shape != (12,):
            raise ValueError(
                f"Expected qpos [12], got {qpos.shape}"
            )

        qpos = torch.from_numpy(
            qpos
        ).to(
            self.device
        )

        self.qpos = (
            (
                qpos
                - self.qpos_mean
            )
            / self.qpos_std
        ).unsqueeze(0)

        # ----------------------------------------------------
        # 2. Head PointCloud
        #
        # RoboTwin observation is World XYZ.
        # Convert to the same Task frame/crop used in training.
        # ----------------------------------------------------
        vision = obs["vision"]

        pointcloud_world = np.asarray(
            vision["pointcloud"],
            dtype=np.float32,
        )

        if (
            pointcloud_world.ndim != 2
            or pointcloud_world.shape[1] < 3
        ):
            raise ValueError(
                "Expected pointcloud [N,>=3], "
                f"got {pointcloud_world.shape}"
            )

        pointcloud_world = (
            pointcloud_world[:, :3]
        )

        finite = np.isfinite(
            pointcloud_world
        ).all(axis=1)

        pointcloud_world = (
            pointcloud_world[finite]
        )

        processed = transform_and_crop(
            pointcloud_world,
            self.task_frame_cfg,
        )

        pointcloud_task = np.asarray(
            processed["cropped"],
            dtype=np.float32,
        )

        if len(pointcloud_task) == 0:
            raise RuntimeError(
                "Task-frame PointCloud crop "
                "contains zero points"
            )

        self.pointcloud = torch.from_numpy(
            np.ascontiguousarray(
                pointcloud_task
            )
        ).to(
            self.device
        )

        # ----------------------------------------------------
        # 3. Left / Right wrist RGB
        # ----------------------------------------------------
        if "cam_left_wrist" not in vision:
            raise KeyError(
                "vision['cam_left_wrist'] missing"
            )

        if "cam_right_wrist" not in vision:
            raise KeyError(
                "vision['cam_right_wrist'] missing"
            )

        left_camera = vision[
            "cam_left_wrist"
        ]
        right_camera = vision[
            "cam_right_wrist"
        ]

        if "color" not in left_camera:
            raise KeyError(
                "cam_left_wrist['color'] missing"
            )

        if "color" not in right_camera:
            raise KeyError(
                "cam_right_wrist['color'] missing"
            )

        self.left_wrist = (
            rgb_to_tensor(
                left_camera["color"]
            )
            .to(self.device)
        )

        self.right_wrist = (
            rgb_to_tensor(
                right_camera["color"]
            )
            .to(self.device)
        )

        # Training used native 240x320 wrist images.
        expected_hw = (240, 320)

        left_hw = tuple(
            self.left_wrist.shape[-2:]
        )
        right_hw = tuple(
            self.right_wrist.shape[-2:]
        )

        if left_hw != expected_hw:
            raise ValueError(
                "Left wrist resolution mismatch: "
                f"expected {expected_hw}, "
                f"got {left_hw}"
            )

        if right_hw != expected_hw:
            raise ValueError(
                "Right wrist resolution mismatch: "
                f"expected {expected_hw}, "
                f"got {right_hw}"
            )

        if not self._debug_printed:
            print(
                "[DAPIER_ACT OBS]",
                f"world_pcd={pointcloud_world.shape}",
                f"crop_pcd={pointcloud_task.shape}",
                f"qpos={tuple(self.qpos.shape)}",
                f"left={tuple(self.left_wrist.shape)}",
                f"right={tuple(self.right_wrist.shape)}",
            )

            print(
                "[DAPIER_ACT OBS] crop xyz:",
                "min=",
                pointcloud_task.min(axis=0).tolist(),
                "max=",
                pointcloud_task.max(axis=0).tolist(),
            )

            self._debug_printed = True

    @torch.no_grad()
    def get_action(self):
        if (
            self.qpos is None
            or self.pointcloud is None
            or self.left_wrist is None
            or self.right_wrist is None
        ):
            raise RuntimeError(
                "update_obs() must be called "
                "before get_action()"
            )

        prediction, _, _ = self.model(
            qpos=self.qpos,

            # Model expects a list of [N,3] clouds.
            pointcloud=[
                self.pointcloud
            ],

            left_wrist=self.left_wrist,
            right_wrist=self.right_wrist,

            # CVAE inference => z = 0
            actions=None,
            is_pad=None,
        )

        if not torch.isfinite(
            prediction
        ).all():
            raise RuntimeError(
                "Policy prediction contains NaN/Inf"
            )

        # Model predicts normalized action.
        prediction = (
            prediction[0]
            * self.action_std
            + self.action_mean
        )

        action_chunk = (
            prediction
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if action_chunk.shape != (
            self.chunk_size,
            12,
        ):
            raise RuntimeError(
                "Unexpected action chunk shape: "
                f"{action_chunk.shape}"
            )

        # ----------------------------------------------------
        # Temporal aggregation
        #
        # Every inference produces:
        #     [chunk_size, 12]
        #
        # At global timestep t, collect all predictions from
        # previous chunks that refer to this same timestep.
        # Newer predictions receive slightly larger weights.
        # ----------------------------------------------------
        t = self.temporal_step

        self.temporal_history.append(
            (
                t,
                action_chunk.copy(),
            )
        )

        # Anything older than the ACT chunk horizon can no
        # longer contribute to the current timestep.
        self.temporal_history = [
            (start_t, chunk)
            for start_t, chunk
            in self.temporal_history
            if 0 <= (t - start_t) < self.chunk_size
        ]

        predictions = []

        # Temporal ensemble:
        # newest prediction -> oldest prediction.
        # With w_i = exp(-k*i), newer predictions receive
        # larger weights.
        for start_t, chunk in self.temporal_history:
            offset = t - start_t

            if 0 <= offset < chunk.shape[0]:
                predictions.append(
                    chunk[offset]
                )

        if not predictions:
            raise RuntimeError(
                "Temporal aggregation has no valid predictions"
            )

        predictions = np.stack(
            predictions,
            axis=0,
        ).astype(np.float32)

        weights = np.exp(
            -self.temporal_k
            * np.arange(
                len(predictions),
                dtype=np.float32,
            )
        )

        weights = (
            weights
            / np.sum(weights)
        )

        # ACT temporal aggregation:
        # combine predictions from overlapping action chunks
        # that correspond to the current timestep.
        aggregated = np.sum(
            predictions * weights[:, None],
            axis=0,
        ).astype(np.float32)

        self.temporal_step += 1

        if self.temporal_step <= 5:
            print(
                "[DAPIER_ACT TEMPORAL]",
                f"step={t}",
                f"predictions={len(predictions)}",
                f"weights={weights.round(4).tolist()}",
            )

        # Keep the same API as before:
        # one-action chunk [1,12].
        aggregated_chunk = (
            aggregated[None, :]
        )

        actions = unpack_robot_state(
            aggregated_chunk,
            action_type="joint",
            robot_action_dim_info=(
                self.robot_action_dim_info
            ),
            source_type="obs",
        )

        return actions

    def update_obs_batch(
        self,
        obs_list,
    ):
        raise NotImplementedError(
            "DAPIER_ACT batch eval is not enabled"
        )

    def get_action_batch(
        self,
        env_idx_list=None,
    ):
        raise NotImplementedError(
            "DAPIER_ACT batch eval is not enabled"
        )

    def reset(self):
        self.qpos = None
        self.pointcloud = None
        self.left_wrist = None
        self.right_wrist = None
        self._debug_printed = False
