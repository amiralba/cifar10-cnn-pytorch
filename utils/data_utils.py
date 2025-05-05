from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_loaders(data_dir="data", batch_size=32, num_workers=4):
    """
    Load CIFAR-10 dataset with specified transformations.
    Args:
        data_dir (str): Directory to store the dataset.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
    Returns:
        train_loader (DataLoader): DataLoader for training set.
        test_loader (DataLoader): DataLoader for test set.
        class_names (list): List of class names in the dataset.
    """
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_data = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    class_names = train_data.classes

    return train_loader, test_loader, class_names
