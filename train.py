import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from tqdm.auto import tqdm
from models.cnn import TinyVGG
from utils.data_utils import get_cifar10_loaders
from models.resnet import load_resnet18

def setup_training(device, hidden_units, output_shape, learning_rate, model_name="tinyvgg"):
    """
    Set up the model, loss function, and optimizer for training.

    Args:
        device (str): Device to use for training (e.g., "cuda" or "cpu").
        hidden_units (int): Number of filters in conv layers.
        learning_rate (float): Learning rate for the optimizer.
        output_shape (int): Number of output classes (e.g., 10 for CIFAR-10).
        model_name (str): Name of the model to use ("tinyvgg" or "resnet18").
    Returns:
        model (nn.Module): The model to be trained.
        loss_fn (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
    """
    if model_name == "resnet18":
        model = load_resnet18(num_classes=output_shape, pretrained=False).to(device)
    else:
        model = TinyVGG(input_shape=3,
                        hidden_units=hidden_units,
                        output_shape=output_shape).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_fn, optimizer

def train_model(model, train_loader, loss_fn, optimizer, device, epochs):
    """
    Trains a PyTorch model and prints training loss and accuracy per epoch.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): Training data loader.
        loss_fn (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (str): Device for model computation ('cuda' or 'cpu').
        epochs (int): Number of epochs to train the model.
    """
    model.to(device)
    print("Training started...\n")
    start = timer()

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for batch, (X, y) in enumerate(loop):
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # 2. Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3. Calculate accuracy
            y_pred_class = torch.argmax(y_pred, dim=1)
            train_correct += (y_pred_class == y).sum().item()

            # 4. Update progress bar
            loop.set_postfix(loss=loss.item())

        accuracy = train_correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Accuracy: {accuracy:.4f}")

    end = timer()
    print(f"\n✅ Training complete in {end - start:.2f} seconds.")
def main():
    # Configuration
    NUM_EPOCHS = 3
    BATCH_SIZE = 32
    HIDDEN_UNITS = 64
    LEARNING_RATE = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_loader, test_loader, class_names = get_cifar10_loaders(batch_size=BATCH_SIZE)

    # Setup model, loss, optimizer
    model, loss_fn, optimizer = setup_training(
        device=device,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names),
        learning_rate=LEARNING_RATE,
        model_name="resnet18"  # 👈 switch this
    )

    # Train the model
    train_model(model, train_loader, loss_fn, optimizer, device, epochs=NUM_EPOCHS)

    # Save the model based on its type
    if isinstance(model, TinyVGG):
        MODEL_PATH = "outputs/tinyvgg_cifar10.pth"
    else:
        MODEL_PATH = "outputs/resnet18_cifar10.pth"

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()