from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
# from sklearn.model_selection import train_test_split
from config import data_path, batch_size
import os
import torch 

def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(split_train=True, train_size=0.8, val_size=0.2):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'images'), transform=transform)

    # Split dataset into train and test sets
    torch.manual_seed(42)
    train_size = int(len(full_dataset) * train_size)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Optionally split the train set into training and validation sets
    data_loaders = {}
    if split_train:
        train_size = int(len(train_dataset) * (1 - val_size))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        data_loaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create DataLoaders for each set
    data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return data_loaders

def save_model(model, path):
    torch.save(model.state_dict(), path)
