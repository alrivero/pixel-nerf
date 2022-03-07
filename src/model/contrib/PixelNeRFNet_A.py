from ..models import PixelNeRFNet
from .AppearanceEncoder import AppearanceEncoder

class PixelNeRFNet_A(PixelNeRFNet):
    # For now, identical to the PixelNeRF
    def __init__(self, conf, stop_encoder_grad=False, stop_app_encoder_grad=False):
        super().__init__(conf, stop_encoder_grad)

        self.use_app_encoder = conf.get_bool("use_app_encoder", True)
        if self.use_app_encoder:
            self.stop_app_encoder_grad = stop_app_encoder_grad
            self.app_encoder = AppearanceEncoder(conf["app_encoder"])
