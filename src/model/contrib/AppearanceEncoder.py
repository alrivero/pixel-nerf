from torch import nn
from .unet_tile_se_norm import StyleEncoder

class AppearanceEncoder(StyleEncoder):
    # Identical to the style encoder of SSHarmonization, but loaded from config file
    def __init__(self, conf):
        n_downsample = conf.get_float("n_downsample", 4)
        input_dim = conf.get_float("input_dim", 3)
        dim = conf.get_float("dim", 512)
        style_dim = conf.get_float("style_dim", 512)

        norm_str = conf.get_string("norm")
        if norm_str == "instance":
            norm = nn.InstanceNorm2d(dim)
        elif norm_str == "batch":
            norm = nn.BatchNorm2d(dim)
        else:
            raise AttributeError("No valid norm assignable to appearance encoder") 

        activ = conf.get_string("activ", "relu")
        pad_type = conf.get_string("pad_type", "reflect")

        super().__init__(n_downsample, input_dim, dim, style_dim, norm, activ, pad_type)