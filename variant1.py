import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from main import data_loading

class CNNModel(nn.Module):
    def __init__(self, num_of_classes):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)  # Increased the number of neurons in the fully connected layer
        self.fc2 = nn.Linear(256, num_of_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluate_model(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = data_loading()
    print(
        f'Data Loaded: Train Images:{len(train_loader.dataset)}, Validation Images:{len(val_loader.dataset)}, Test Images: {len(test_loader.dataset)}')

    model = CNNModel(4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 10

    for epoch in range(num_epochs):
        running_loss = 0
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        if (epoch + 1) % 5 == 0:
            accuracy, avg_val_loss = evaluate_model(model, val_loader, criterion, device)
            print(f'After: {epoch + 1} epochs')
            print(f'Validation Accuracy: {accuracy:.2f}%')
            print(f'Average Validation Loss: {avg_val_loss:.4f}')

            if accuracy > 80:
                break

    torch.save(model.state_dict(), 'model_trained2.pth')
    print('Model Saved')
