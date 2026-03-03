 


"""Configuration for the ray-cast sensor."""


from isaaclab.sensors.ray_caster import RayCasterCfg as BaseRayCasterCfg
from isaaclab.utils import configclass

from .ray_caster import RayCaster


@configclass
class RayCasterCfg(BaseRayCasterCfg):

    class_type: type = RayCaster
