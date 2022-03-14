from torch import nn
from .unet_tile_se_norm import StyleEncoder

class AppearanceEncoder(StyleEncoder):
    # Identical to the style encoder of SSHarmonization, but loaded from config file
    def __init__(self, conf):
        n_downsample = conf.get_int("n_downsample", 4)
        input_dim = conf.get_int("input_dim", 3)
        dim = conf.get_int("dim", 512)
        style_dim = conf.get_int("style_dim", 512)
        norm = conf.get_string("norm", "none")
        activ = conf.get_string("activ", "relu")
        pad_type = conf.get_string("pad_type", "reflect")

        super().__init__(n_downsample, input_dim, dim, style_dim, norm, activ, pad_type)

        self.app_encoding = None
    
    def encode(self, app_data):
        self.app_encoding = self(app_data["images"])