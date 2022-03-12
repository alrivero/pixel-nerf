import torch
from contrib.model.unet_tile_se_norm import StyleEncoder

class AlphaLossNV2(torch.nn.Module):
    """
    Implement Neural Volumes alpha loss 2
    """

    def __init__(self, lambda_alpha, clamp_alpha, init_epoch, force_opaque=False):
        super().__init__()
        self.lambda_alpha = lambda_alpha
        self.clamp_alpha = clamp_alpha
        self.init_epoch = init_epoch
        self.force_opaque = force_opaque
        if force_opaque:
            self.bceloss = torch.nn.BCELoss()
        self.register_buffer(
            "epoch", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def sched_step(self, num=1):
        self.epoch += num

    def forward(self, alpha_fine):
        if self.lambda_alpha > 0.0 and self.epoch.item() >= self.init_epoch:
            alpha_fine = torch.clamp(alpha_fine, 0.01, 0.99)
            if self.force_opaque:
                alpha_loss = self.lambda_alpha * self.bceloss(
                    alpha_fine, torch.ones_like(alpha_fine)
                )
            else:
                alpha_loss = torch.log(alpha_fine) + torch.log(1.0 - alpha_fine)
                alpha_loss = torch.clamp_min(alpha_loss, -self.clamp_alpha)
                alpha_loss = self.lambda_alpha * alpha_loss.mean()
        else:
            alpha_loss = torch.zeros(1, device=alpha_fine.device)
        return alpha_loss


def get_alpha_loss(conf):
    lambda_alpha = conf.get_float("lambda_alpha")
    clamp_alpha = conf.get_float("clamp_alpha")
    init_epoch = conf.get_int("init_epoch")
    force_opaque = conf.get_bool("force_opaque", False)

    return AlphaLossNV2(
        lambda_alpha, clamp_alpha, init_epoch, force_opaque=force_opaque
    )


class RGBWithUncertainty(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17"""

    def __init__(self, conf):
        super().__init__()
        self.element_loss = (
            torch.nn.L1Loss(reduction="none")
            if conf.get_bool("use_l1")
            else torch.nn.MSELoss(reduction="none")
        )

    def forward(self, outputs, targets, betas):
        """computes the error per output, weights each element by the log variance
        outputs is B x 3, targets is B x 3, betas is B"""
        weighted_element_err = (
            torch.mean(self.element_loss(outputs, targets), -1) / betas
        )
        return torch.mean(weighted_element_err) + torch.mean(torch.log(betas))


class RGBWithBackground(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17"""

    def __init__(self, conf):
        super().__init__()
        self.element_loss = (
            torch.nn.L1Loss(reduction="none")
            if conf.get_bool("use_l1")
            else torch.nn.MSELoss(reduction="none")
        )

    def forward(self, outputs, targets, lambda_bg):
        """If we're using background, then the color is color_fg + lambda_bg * color_bg.
        We want to weight the background rays less, while not putting all alpha on bg"""
        weighted_element_err = torch.mean(self.element_loss(outputs, targets), -1) / (
            1 + lambda_bg
        )
        return torch.mean(weighted_element_err) + torch.mean(torch.log(lambda_bg))

class ReferenceColorLoss(torch.nn.Module):
    """SSH relighting reference encoding comparison"""

    def __init__(self, conf, encoder_dir=None):
        super().__init__()
        self.ref_encoder = StyleEncoder(4, 3, 32, 512, norm="BN", activ="relu", pad_type='reflect')
        self.ref_encoder.eval()
        if conf.get_bool("pretrained") and encoder_dir is not None:
            self.ref_encoder.load_state_dict(torch.load(encoder_dir))
            print("using refernce color loss with pretrained weights")
        else:
            print("using refernce color loss WITHOUT pretrained weights")
        
        for param in self.ref_encoder.parameters():
            param.requires_grad = False
        
        
        self.ref_loss = torch.torch.nn.MSELoss()
    
    def forward(self, outputs, targets):
        outputs_encodings = self.ref_encoder(outputs)
        targets_encodings = self.ref_encoder(targets)

        return self.ref_loss(outputs_encodings, targets_encodings)


def get_rgb_loss(conf, coarse=True, using_bg=False, reduction="mean"):
    if conf.get_bool("use_uncertainty", False) and not coarse:
        print("using rgb loss with uncertainty")
        return RGBWithUncertainty(conf)
    #     if using_bg:
    #         print("using loss with background")
    #         return RGBWithBackground(conf)
    print("using vanilla rgb loss")
    return (
        torch.nn.L1Loss(reduction=reduction)
        if conf.get_bool("use_l1")
        else torch.nn.MSELoss(reduction=reduction)
    )

def get_density_loss(conf, using_bg=False, reduction="mean"):
    print("using vanilla density loss")
    return (
        torch.nn.L1Loss(reduction=reduction)
        if conf.get_bool("use_l1")
        else torch.nn.MSELoss(reduction=reduction)
    )
