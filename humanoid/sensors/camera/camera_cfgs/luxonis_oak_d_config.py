import math

from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from humanoid.sensors.camera import CameraCfg, SensorNoiseCfg, TiledCameraCfg

LUXONOAK_DEPTH_HFOV_DEG = 72.0
LUXONOAK_HORIZONTAL_APERTURE = 2.4
LUXONOAK_FOCAL_LENGTH = (LUXONOAK_HORIZONTAL_APERTURE / 2.0) / math.tan(math.radians(LUXONOAK_DEPTH_HFOV_DEG) / 2.0)


@configclass
class LuxonisOakD:
    enable_depth_camera = True
    debug_vis = False

    width: int = 480
    height: int = 270
    max_range: float = 12.0
    min_range: float = 0.7

    data_types: list[str] = ["distance_to_image_plane"]

    offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
        pos=(0.10, 0.0, 0.03), rot=(0.707, 0.0, 0.707, 0.0), convention="ros"
    )

    spawn: PinholeCameraCfg = PinholeCameraCfg(
        focal_length=LUXONOAK_FOCAL_LENGTH,
        horizontal_aperture=LUXONOAK_HORIZONTAL_APERTURE,
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
class LuxonisOakDConfig(LuxonisOakD, CameraCfg):
    pass


@configclass
class TiledLuxonisOakDConfig(LuxonisOakD, TiledCameraCfg):
    pass
