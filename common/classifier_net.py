"""Classifier architecture shared by training, evaluation, the FastAPI backend and the Flask app.

State-dict layout is `features.*` + `classifier.*` (identical to the original training/model.py for
mobilenet_v3 / resnet18), so previously trained checkpoints stay loadable.

Backbones: mobilenet_v3 | resnet18 | resnet50 | efficientnet_b0 | custom
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

BACKBONES = ("mobilenet_v3", "resnet18", "resnet50", "efficientnet_b0", "custom")


def _try_pretrained(builder, weights_enum, name):
    """Build a torchvision model with ImageNet weights; fall back to random init (loudly)."""
    try:
        return builder(weights=weights_enum.DEFAULT), True
    except Exception as e:  # no internet / cache miss
        print(f"[classifier_net] WARNING: could not load pretrained {name} weights ({e}). "
              f"Falling back to RANDOM initialisation - accuracy will be noticeably lower.")
        return builder(weights=None), False


def _head(in_features: int, num_classes: int, act: nn.Module, dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(in_features, 256),
        act,
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )


class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True, backbone: str = "mobilenet_v3"):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from {BACKBONES}")
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.pretrained_loaded = False

        if backbone == "mobilenet_v3":
            base, ok = (_try_pretrained(tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights, backbone)
                        if pretrained else (tvm.mobilenet_v3_small(weights=None), False))
            self.features = base.features
            self.classifier = _head(base.classifier[0].in_features, num_classes, nn.Hardswish())
        elif backbone in ("resnet18", "resnet50"):
            b, w = (tvm.resnet18, tvm.ResNet18_Weights) if backbone == "resnet18" else (tvm.resnet50, tvm.ResNet50_Weights)
            base, ok = _try_pretrained(b, w, backbone) if pretrained else (b(weights=None), False)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.classifier = _head(base.fc.in_features, num_classes, nn.ReLU(inplace=True))
        elif backbone == "efficientnet_b0":
            base, ok = (_try_pretrained(tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights, backbone)
                        if pretrained else (tvm.efficientnet_b0(weights=None), False))
            self.features = base.features
            self.classifier = _head(base.classifier[1].in_features, num_classes, nn.SiLU())
        else:  # custom small CNN
            ok = False
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            )
            self.classifier = _head(256, num_classes, nn.ReLU(inplace=True))

        self.pretrained_loaded = bool(ok)
        self.target_layer = self.features[-1]

    # ------------------------------------------------------------------ forward
    def forward(self, x):
        return self.classifier(self.features(x))

    # ---- helpers used by the training script (backbone warm-up then full fine-tune)
    def set_backbone_trainable(self, trainable: bool):
        for p in self.features.parameters():
            p.requires_grad = trainable

    def head_parameters(self):
        return self.classifier.parameters()

    def backbone_parameters(self):
        return self.features.parameters()

    # ------------------------------------------------------------------ Grad-CAM
    def generate_gradcam(self, input_tensor, target_class=None):
        """Grad-CAM for one image (1,3,H,W). Returns (heatmap[0..1], probs(np), target_class).

        Hooks exist only for the duration of this call, so normal training/inference is unaffected.
        """
        was_training = self.training
        self.eval()
        store = {}

        def fwd_hook(_m, _i, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        handle = self.target_layer.register_forward_hook(fwd_hook)
        try:
            self.zero_grad(set_to_none=True)
            with torch.enable_grad():
                x = input_tensor.clone().requires_grad_(True)
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                if target_class is None:
                    target_class = int(torch.argmax(probs, dim=1).item())
                logits[0, target_class].backward()
        finally:
            handle.remove()
            if was_training:
                self.train()

        probs_np = probs[0].detach().cpu().numpy()
        if "act" not in store or "grad" not in store:
            return None, probs_np, target_class

        weights = store["grad"].mean(dim=(2, 3), keepdim=True)            # (1,C,1,1)
        cam = F.relu((weights * store["act"].detach()).sum(dim=1))[0]      # (h,w)
        cam = cam.cpu().numpy()
        mx = cam.max()
        cam = cam / mx if mx > 1e-8 else np.zeros_like(cam)
        return cam, probs_np, target_class
