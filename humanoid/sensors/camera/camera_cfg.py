from dataclasses import dataclass
from typing import Literal

import torch
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors.camera import CameraCfg as BaseCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sim.spawners import PreviewSurfaceCfg, SphereCfg
from isaaclab.utils import configclass

from .camera import Camera


@dataclass
class SensorNoiseCfg:
    """Configuration for sensor noise."""

    enable: bool = False
    mode: Literal["gaussian", "dropout", "combined"] = "gaussian"

    # Gaussian noise parameters
    depth_std: float = 0.01
    depth_std_multiplier: float = 0.01

    # Dropout noise parameters
    dropout_prob: float = 0.01
    dropout_value: float = 0.0


@configclass
class CameraCfg(BaseCameraCfg):
    class_type: type = Camera

    enable_depth_camera: bool = False
    prim_body_name: str = "pelvis/depth_camera"

    # Camera parameters
    width: int = 480
    height: int = 270
    max_range: float = 15.0
    min_range: float = 0.2

    data_types: list[str] = ["distance_to_image_plane"]
    offset: BaseCameraCfg.OffsetCfg = BaseCameraCfg.OffsetCfg()
    spawn: PinholeCameraCfg = PinholeCameraCfg()
    sensor_noise: SensorNoiseCfg = SensorNoiseCfg()

    # Camera Visualization Configuration
    debug_vis: bool = False
    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/CameraPointCloud",
        markers={
            "point": SphereCfg(
                radius=0.02,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            )
        },
    )
    visualizer_cfg.decimation = 10

    far_out_of_range_value = torch.inf
    near_out_of_range_value = torch.inf
