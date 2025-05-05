import torch
import torch.nn as nn


class TinyVGG(nn.Module):
    """
    A VGG-inspired CNN architecture using two convolutional blocks.

    Args:
        input_shape (int): Number of input channels (e.g., 3 for RGB)
        hidden_units (int): Number of filters in conv layers
        output_shape (int): Number of output classes (e.g., 10 for CIFAR-10)

    Architecture:
        ConvBlock1: Conv → ReLU → Conv → ReLU → MaxPool
        ConvBlock2: Conv → ReLU → Conv → ReLU → MaxPool
        Classifier: Flatten → Linear
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 5 * 5,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input image tensor (B, C, H, W)

        Returns:
            Tensor: Output logits for classification (B, output_shape)
        """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x