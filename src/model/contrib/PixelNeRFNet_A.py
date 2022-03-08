from ..models import PixelNeRFNet
from .AppearanceEncoder import AppearanceEncoder

class PixelNeRFNet_A(PixelNeRFNet):
    # For now, identical to the PixelNeRF
    def __init__(self, conf, stop_encoder_grad=False, stop_app_encoder_grad=False):
        super().__init__(conf, stop_encoder_grad)

        self.use_app_encoder = conf.get_bool("use_app_encoder", True)
        self.stop_app_encoder_grad = stop_app_encoder_grad
        if self.use_app_encoder:
            self.app_encoder = AppearanceEncoder(conf["app_encoder"])
    
    def load_weights(self, args, opt_init=False, strict=False, device=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/pixel_nerf_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        self = super().load_weights(self, args, opt_init, strict, device)

        # Only load weights for our appearance encoder if we want to
        if args.load_app_encoder:
            return

        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, "app_encoder_init")

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load (Appearance Encoder)", model_path)
            self.app_encoder.load_state_dict(
                torch.load(model_path, map_location=device), strict=True
            )
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Apearance encoder will be re-initialized.\n"
                    + "If you are trying to load a pretrained appearance encoder for PixelNeRF-A, STOP since it's not in the right place. "
                ).format(model_path)
            )
        return self