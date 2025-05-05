from torchvision.models import resnet18
import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

def load_resnet18(num_classes: int = 10, pretrained: bool = False):
    """
    Loads a ResNet18 model and modifies the final layer for custom classification.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load ImageNet-pretrained weights.

    Returns:
        nn.Module: Modified ResNet18 model.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model