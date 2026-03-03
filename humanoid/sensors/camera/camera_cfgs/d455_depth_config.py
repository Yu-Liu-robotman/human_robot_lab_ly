import math

from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from humanoid.sensors.camera import CameraCfg, SensorNoiseCfg, TiledCameraCfg

D455_DEPTH_HFOV_DEG = 87.0
D455_HORIZONTAL_APERTURE_CM = 2.4
D455_FOCAL_LENGTH_CM = (D455_HORIZONTAL_APERTURE_CM / 2.0) / math.tan(math.radians(D455_DEPTH_HFOV_DEG) / 2.0)


@configclass
class D455Camera:
    enable_depth_camera = True
    debug_vis = False

    width: int = 480
    height: int = 270
    max_range: float = 15.0
    min_range: float = 0.2

    data_types: list[str] = ["distance_to_image_plane"]

    offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
        pos=(0.10, 0.0, 0.03), rot=(0.707, 0.0, 0.707, 0.0), convention="ros"
    )

    spawn: PinholeCameraCfg = PinholeCameraCfg(
        focal_length=D455_FOCAL_LENGTH_CM,
        horizontal_aperture=D455_HORIZONTAL_APERTURE_CM,
        clipping_range=(min_range, max_range),
    )

    sensor_noise: SensorNoiseCfg = SensorNoiseCfg(
        enable=True,
        mode="combined",
        depth_std=0.005,
        depth_std_multiplier=0.01,
        dropout_prob=0.01,
        dropout_value=-1.0,
    )


@configclass
class D455CameraCfg(D455Camera, CameraCfg):
    pass


@configclass
class TiledD455CameraCfg(D455Camera, TiledCameraCfg):
    pass
