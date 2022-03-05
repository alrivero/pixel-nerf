from ..pixel_nerf.src.model.models import PixelNeRFNet

class PixelNeRFNet_A(PixelNeRFNet):
    # For now, identical to the PixelNeRF
    def __init__(self, conf, stop_encoder_grad=False):
        print("AMONG US\nAMOGUS\n")
        super().__init__(conf, stop_encoder_grad)