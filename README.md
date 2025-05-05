# 🧠 CIFAR-10 Image Classification with TinyVGG & ResNet18 (PyTorch)

This project trains and evaluates two convolutional neural networks — a custom TinyVGG and a ResNet18 — on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset using PyTorch.

## 📦 Project Structure

```
├── models/                 # Model definitions (TinyVGG, ResNet18)
│   ├── cnn.py
│   └── resnet.py
├── utils/                  # Data loading utilities
│   └── data_utils.py
├── logs/                   # Evaluation outputs and visualizations
│   ├── confusion_matrix.png
│   └── example_predictions.png
├── outputs/                # Saved model weights
│   ├── tinyvgg_cifar10.pth
│   └── resnet18_cifar10.pth
├── train.py                # Training script
├── test.py                 # Evaluation and visualization script
├── config.yaml             # Configuration (optional)
├── README.md               # Project README
```

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install torch torchvision matplotlib torchmetrics seaborn scikit-learn
```

### 2. Train the Model

Edit `train.py` to choose which model to train (`tinyvgg` or `resnet18`):

```python
# Example: train ResNet18
setup_training(..., model_name="resnet18")
```

Then run:
```bash
python train.py
```

The model will be saved to either `outputs/tinyvgg_cifar10.pth` or `outputs/resnet18_cifar10.pth`.

### 3. Evaluate and Visualize

```bash
python test.py
```

This will:
- Load the appropriate model
- Calculate accuracy
- Save a confusion matrix plot to `logs/confusion_matrix.png`
- Save 8 example predictions to `logs/example_predictions.png`

## 📊 Results

- ✅ Test Accuracy (TinyVGG): ~X.XX
- ✅ Test Accuracy (ResNet18): ~X.XX
- 🖼️ [Confusion Matrix](logs/confusion_matrix.png)
- 📷 [Example Predictions](logs/example_predictions.png)

## 📚 Model Architectures

### TinyVGG
- 2 Convolutional Blocks (Conv → ReLU → Conv → ReLU → MaxPool)
- Flatten → Fully Connected Layer

### ResNet18
- Deep CNN with residual (skip) connections
- 4 residual stages + global average pooling → Fully Connected Layer
- Based on the original [ResNet paper (2015)](https://arxiv.org/abs/1512.03385)

## ✍️ Author

Built with ❤️ by [Amir Tabatabaei](https://www.linkedin.com/in/amirtabatabaei10/)
