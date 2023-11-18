import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split


def data_loading(data_folder='new_dataset/'):
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset and split into training, validation, and test sets
    data = datasets.ImageFolder(data_folder, transform=transform)
    train_ratio, val_ratio = 0.7, 0.15
    total_samples = len(data)

    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)

    train_data, temp_data = train_test_split(data, train_size=train_samples, random_state=42)
    test_data, validation_data = train_test_split(temp_data, test_size=val_samples, random_state=42)

    # Create DataLoader instances for training, validation, and test sets
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    return train_loader, val_loader, test_loader


class CNNModel(nn.Module):
    def __init__(self, num_of_classes):
        super(CNNModel, self).__init__()

        # Define the architecture of the convolutional neural network
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)
        self.fc2 = nn.Linear(10, num_of_classes)

    def forward(self, x):
        # Define the forward pass of the model
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Removed relu for better performance (especially for multi-class classification)
        return x


def evaluate_model(model, loader, criterion):
    # Evaluate the model on a given data loader
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(loader)

    return accuracy, avg_loss


if __name__ == "__main__":
    # Load data loaders for training, validation, and test sets
    train_loader, val_loader, test_loader = data_loading()
    print(
        f'Data Loaded: Train Images:{len(train_loader.dataset)}, Validation Images:{len(val_loader.dataset)}, Test Images: {len(test_loader.dataset)}')

    # Initialize the model, loss function, and optimizer
    model = CNNModel(4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 10

    # Train the model for a specified number of epochs
    for epoch in range(num_epochs):
        running_loss = 0
        model.train()

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        # Evaluate the model on the validation set every 10 epochs
        if (epoch + 1) % 10 == 0:
            accuracy, avg_val_loss = evaluate_model(model, val_loader, criterion)
            print(f'After: {epoch + 1} epochs')
            print(f'Validation Accuracy: {accuracy:.2f}%')
            print(f'Average Validation Loss: {avg_val_loss:.4f}')

            # Break the training loop if the validation accuracy exceeds 80%
            if accuracy > 80:
                break

    # Save the trained model
    torch.save(model.state_dict(), 'model_trained')
    print('Main Model Saved')
