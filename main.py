import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

def dataLoading():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load data from folders
    data = datasets.ImageFolder('new_dataset/', transform=transform)

    train_ratio = 0.7
    val_ratio = 0.15
    total_samples = len(data)

    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)

    train_data, temp_data = train_test_split(data, train_size=train_samples, random_state=42)
    test_data, validation_data = train_test_split(temp_data, test_size=val_samples, random_state=42)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return train_loader, val_loader, test_loader

class CNNModel(nn.Module):

    def __init__(self, num_of_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)
        self.fc2 = nn.Linear(10, num_of_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


def evaluate_model(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient computation during validation
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

    train_loader, val_loader, test_loader = dataLoading()
    print(f'Data Loaded: Train Images:{len(train_loader)*32}, Validation Images:{len(val_loader)*32} ,Test Images: {len(test_loader)*32}')

    model = CNNModel(4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 30

    for epoch in range(num_epochs):

        if (epoch + 1) % 10 == 0:
            accuracy, avg_val_loss = evaluate_model(model, val_loader, criterion)
            print(f'After: {epoch + 1} epoch')
            print(f'Validation Accuracy: {accuracy:.2f}%')
            print(f'Average Validation Loss: {avg_val_loss:.4f}')
            if accuracy > 80:
                break

        running_loss = 0
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(running_loss / len(train_loader))


    torch.save(model.state_dict(), 'model_trained')
    print('Model Saved')






