import torch
from torch import nn
from .unet_tile_se_norm import Conv2dBlock, SELayer

class PatchEncoder(nn.Module):
    def __init__(self, conf):
        super(PatchEncoder, self).__init__()

        n_downsample = conf.get_int("n_downsample", 2)
        input_dim = conf.get_int("input_dim", 3)
        out_dim = conf.get_int("out_dim", 32)
        activ = conf.get_string("activ", "relu")

        inter_dim = out_dim // (2 ** n_downsample)
        self.init_layer = Conv2dBlock(input_dim, inter_dim, 7, 1, activation=activ)

        self.down_layers = []
        next_inter_dim = inter_dim
        for _ in range(n_downsample):
            next_inter_dim *= 2
            self.down_layers.append(Conv2dBlock(inter_dim, next_inter_dim, 3, 1, activation=activ))
            inter_dim *= 2
        self.down_layers = nn.Sequential(*self.down_layers)

        self.se_layer = SELayer(inter_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, patches):
        x = self.init_layer(patches)
        x = self.down_layers(x)

        x = self.se_layer(x)
        out  = self.avg_pool(x)
        return out.flatten(2)