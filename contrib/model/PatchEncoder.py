from torch import nn

class PatchEncoder(nn.Module):
    def __init__(self, ref_enc):
        super(PatchEncoder, self).__init__()

        self.ref_encoder = ref_enc
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, patches):
        x = self.ref_encoder(patches)
        return self.avg_pool(x)