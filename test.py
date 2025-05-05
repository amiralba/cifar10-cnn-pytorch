import torch
from models.cnn import TinyVGG
from utils.data_utils import get_cifar10_loaders
from torchmetrics.classification import MulticlassAccuracy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torchvision
from models.resnet import load_resnet18

def evaluate_model(model_fn=load_resnet18, model_path="outputs/tinyvgg_cifar10.pth", batch_size=32, **model_kwargs):    
    """
    Loads a trained model and evaluates it on the CIFAR-10 test set.

    Args:
        model_path (str): Path to the saved model weights.
        hidden_units (int): Number of hidden units in the CNN model.
        batch_size (int): Batch size for test data loader.

    Returns:
        Tuple[float, list, list, list]: Final test accuracy, all predictions, all labels, class names
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    _, test_loader, class_names = get_cifar10_loaders(batch_size=batch_size)

    model = model_fn(num_classes=len(class_names), **model_kwargs)    
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()

    accuracy_metric = MulticlassAccuracy(num_classes=len(class_names), average="micro").to(DEVICE)

    all_preds = []
    all_labels = []
    sample_images = []
    sample_preds = []
    sample_true = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            accuracy_metric.update(preds, labels)

            if len(sample_images) < 8:
                sample_images.extend(images[:8].cpu())
                sample_preds.extend(preds[:8].cpu())
                sample_true.extend(labels[:8].cpu())

    final_accuracy = accuracy_metric.compute().item()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    print(f"\n📊 Test Accuracy: {final_accuracy:.4f}")
    return final_accuracy, all_preds, all_labels, class_names, sample_images, sample_preds, sample_true


def plot_confusion_matrix(preds, labels, class_names, save_path="logs/confusion_matrix.png"):
    """
    Plots and saves the confusion matrix for predictions vs actual labels.

    Args:
        preds (list): Predicted class labels.
        labels (list): Ground truth class labels.
        class_names (list): List of class names.
        save_path (str): Path to save the confusion matrix image.
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📁 Saved confusion matrix to: {save_path}")
    plt.show()


def show_example_predictions(images, preds, true, class_names, save_path="logs/example_predictions.png"):
    """
    Displays and saves a batch of images with their predicted and true labels.

    Args:
        images (list): List of input images (tensors).
        preds (list): List of predicted class indices.
        true (list): List of true class indices.
        class_names (list): List of class names.
        save_path (str): Path to save the prediction examples image.
    """
    plt.figure(figsize=(12, 6))
    for idx in range(min(len(images), 8)):
        img = images[idx]
        img = (img * 0.5) + 0.5  # unnormalize
        img = np.transpose(img.numpy(), (1, 2, 0))

        plt.subplot(2, 4, idx + 1)
        plt.imshow(img)
        plt.title(f"Pred: {class_names[preds[idx]]}\nTrue: {class_names[true[idx]]}")
        plt.axis("off")
    plt.suptitle("Example Predictions")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📁 Saved example predictions to: {save_path}")
    plt.show()


def main():
    # Select the model function
    model_fn = load_resnet18
    model_path = "outputs/resnet18_cifar10.pth" if model_fn == load_resnet18 else "outputs/tinyvgg_cifar10.pth"

    # Evaluate
    acc, preds, labels, class_names, sample_images, sample_preds, sample_true = evaluate_model(
        model_fn=model_fn,
        model_path=model_path,
        batch_size=32,
        pretrained=False  # Set to True if evaluating a pretrained model fine-tuned on CIFAR-10
    )

    # Save to logs directory
    plot_confusion_matrix(preds, labels, class_names, save_path="logs/confusion_matrix.png")
    show_example_predictions(sample_images, sample_preds, sample_true, class_names, save_path="logs/example_predictions.png")

if __name__ == "__main__":
    main()