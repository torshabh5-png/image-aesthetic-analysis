
"""
model.py
========
MobileNetV3-Small fine-tuned for image aesthetic scoring.

Supports:
1. Regression       -> Predict aesthetic score
2. Classification   -> Predict Low / Medium / High class
"""

import torch
import torch.nn as nn
from torchvision import models


class AestheticMobileNet(nn.Module):
    """
    MobileNetV3-Small backbone with custom prediction head.

    Parameters
    ----------
    task : str
        'regression' or 'classification'

    pretrained : bool
        Load ImageNet pretrained weights

    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        task: str = "regression",
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()

        # -------------------------------
        # Validate task
        # -------------------------------
        if task not in ["regression", "classification"]:
            raise ValueError(
                "task must be either 'regression' or 'classification'"
            )

        self.task = task

        # -------------------------------
        # Load MobileNetV3-Small
        # -------------------------------
        weights = (
            models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        backbone = models.mobilenet_v3_small(weights=weights)

        # Feature extractor
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Input features for classifier
        in_features = backbone.classifier[0].in_features

        # Output neurons
        out_features = 1 if task == "regression" else 3

        # -------------------------------
        # Custom prediction head
        # -------------------------------
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.Hardswish(),
            nn.Dropout(dropout / 2),

            nn.Linear(64, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """

        # Backbone
        x = self.features(x)

        # Global average pooling
        x = self.avgpool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Prediction head
        x = self.head(x)

        # Regression output shape: (B,)
        if self.task == "regression":
            x = x.squeeze(1)

        return x


def build_model(
    task: str = "regression",
    pretrained: bool = True,
    dropout: float = 0.3,
    device: str = "auto"
) -> AestheticMobileNet:
    """
    Convenience function to create model and move to device.
    """

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AestheticMobileNet(
        task=task,
        pretrained=pretrained,
        dropout=dropout
    )

    model = model.to(device)

    # Count trainable parameters
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("\n" + "=" * 60)
    print(f"Model            : AestheticMobileNet")
    print(f"Task             : {task}")
    print(f"Device           : {device}")
    print(f"Trainable Params : {trainable_params:,}")
    print("=" * 60 + "\n")

    return model


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy input
    dummy_input = torch.randn(4, 3, 224, 224).to(DEVICE)

    # --------------------------------------------------------
    # Regression Test
    # --------------------------------------------------------
    print("Testing Regression Model...\n")

    reg_model = build_model(
        task="regression",
        pretrained=False,
        device=DEVICE
    )

    reg_output = reg_model(dummy_input)

    print("Regression Output Shape :", reg_output.shape)
    print("Regression Output       :", reg_output)
    print()

    # --------------------------------------------------------
    # Classification Test
    # --------------------------------------------------------
    print("Testing Classification Model...\n")

    cls_model = build_model(
        task="classification",
        pretrained=False,
        device=DEVICE
    )

    cls_output = cls_model(dummy_input)

    print("Classification Output Shape :", cls_output.shape)
    print("Classification Output       :")
    print(cls_output)

