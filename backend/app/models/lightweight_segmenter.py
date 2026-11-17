import os
import torch
import torch.nn as nn

from backend.app import config
from backend.app.models.segmenter import ModelUnavailableError, load_state_dict_strict


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class FastBrainUNet(nn.Module):
    """
    Lightweight, pruned segmentation network designed for edge / low-latency clinical inference.
    Uses depthwise separable convolutions and reduced channel widths (~1.1M params).
    Dropout2d layers (parameter-free) make Monte-Carlo-dropout uncertainty available for this model too.
    """
    def __init__(self, in_channels=3, num_classes=1, mc_dropout=0.1):
        super().__init__()
        self.enc1 = nn.Sequential(DepthwiseSeparableConv(in_channels, 24), DepthwiseSeparableConv(24, 24))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(DepthwiseSeparableConv(24, 48), DepthwiseSeparableConv(48, 48))
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(DepthwiseSeparableConv(48, 96), DepthwiseSeparableConv(96, 96))
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(DepthwiseSeparableConv(96, 160), DepthwiseSeparableConv(160, 160))
        self.drop_b = nn.Dropout2d(mc_dropout)

        self.up3 = nn.ConvTranspose2d(160, 96, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(DepthwiseSeparableConv(192, 96), DepthwiseSeparableConv(96, 96))
        self.drop_d3 = nn.Dropout2d(mc_dropout)

        self.up2 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(DepthwiseSeparableConv(96, 48), DepthwiseSeparableConv(48, 48))

        self.up1 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(DepthwiseSeparableConv(48, 24), DepthwiseSeparableConv(24, 24))

        self.final = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.drop_b(self.bottleneck(self.pool3(e3)))

        d3 = self.drop_d3(self.dec3(torch.cat([self.up3(b), e3], dim=1)))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


_lightweight_cache = {}


def lightweight_weights_available() -> bool:
    return os.path.exists(config.FASTBRAIN_WEIGHTS_PATH)


def get_lightweight_segmenter(device=None, require_weights=True):
    """
    -> (model, device). By default raises ModelUnavailableError when no TRAINED weights exist.
    (Previously this returned a randomly-initialised network, so selecting it produced garbage masks.)
    Train it with:  python training/train_segmentation.py --model fast --teacher models/best_nnunet_model.pt
    """
    from backend.app.models.classifier import resolve_device
    device = resolve_device(device)

    key = "trained" if require_weights else "arch_only"
    if key in _lightweight_cache:
        return _lightweight_cache[key], device

    model = FastBrainUNet(in_channels=3, num_classes=1)
    if require_weights:
        if not lightweight_weights_available():
            raise ModelUnavailableError(
                f"FastBrain-UNet has not been trained yet (no weights at {config.FASTBRAIN_WEIGHTS_PATH}). "
                f"Run training/train_segmentation.py --model fast first.")
        load_state_dict_strict(model, config.FASTBRAIN_WEIGHTS_PATH, device)
        print(f"[NeuroVision AI] Loaded FastBrain-UNet weights from: {config.FASTBRAIN_WEIGHTS_PATH}")

    model = model.to(device)
    model.eval()
    _lightweight_cache[key] = model
    return model, device
