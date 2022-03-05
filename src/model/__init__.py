import sys
from .models import PixelNeRFNet

_ogpath = sys.path[:]
sys.path.append('../../Contrib')
from PixelNeRFNet_A import PixelNeRFNet_A
sys.path = _ogpath

def make_model(conf, *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get_string("type", "pixelnerf")  # single
    if model_type == "pixelnerf":
        net = PixelNeRFNet(conf, *args, **kwargs)
    else if model_type == "pixelnerf-a":
        net = PixelNeRFNet_A(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
