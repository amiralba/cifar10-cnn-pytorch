import torch
from train import mod
torch.save(model.state_dict(), "outputs/tinyvgg_cifar10.pth")
print("✅ Model saved to outputs/tinyvgg_cifar10.pth")