import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

def data_loading(data_folder='new_dataset/'):
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    data = datasets.ImageFolder(data_folder, transform=transform)

    # Create DataLoader instances for full dataset
    data_loader = DataLoader(data, batch_size=32, shuffle=True)
    return data_loader


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

def evaluate_model(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradient computation during validation
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(loader)
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, avg_loss, cm, all_preds, all_labels

if __name__ == "__main__":
    # Load data loaders for training, validation, and test sets
    data_loader = data_loading()
    full_dataset_size = len(data_loader.dataset)
    print('Full DataSet loaded')

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folds = 10
    kf = KFold(n_splits=10,shuffle=True,random_state=42)
    Acc=0
    num_epochs = 10
    model = CNNModel(4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for fold, (train_index, test_index) in enumerate(kf.split(range(full_dataset_size))):

        train_loader = DataLoader(torch.utils.data.Subset(data_loader.dataset, train_index), shuffle=True)
        test_loader = DataLoader(torch.utils.data.Subset(data_loader.dataset, test_index))

        for epoch in range(num_epochs):

            model.train()
            running_loss = 0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
            accuracy, avg_test_loss, cm, y_pred, y_test = evaluate_model(model,test_loader,criterion, device)

            if accuracy>Acc:
                Acc=accuracy
                torch.save(model.state_dict(), 'model_trained_KFold')


        print(f'After Fold: {fold + 1}')
        print(f'Test Accuracy: {accuracy:.2f}% on fold- {fold+1}')
        print(f'Average Test Loss: {avg_test_loss:.4f} on fold- {fold+1}')
        # Calculate additional metrics
        acc = accuracy_score(y_test, y_pred)
        pre_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        pre_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

        # Print additional metrics
        print(f'Test Accuracy: {acc * 100:.2f}%')
        print(f'Test Precision (Macro): {pre_macro * 100:.2f}%')
        print(f'Test Precision (Micro): {pre_micro * 100:.2f}%')
        print(f'Test Recall (Macro): {recall_macro * 100:.2f}%')
        print(f'Test Recall (Micro): {recall_micro * 100:.2f}%')
        print(f'Test F1 Score (Macro): {f1_macro * 100:.2f}%')
        print(f'Test F1 Score (Micro): {f1_micro * 100:.2f}%')




# Save the trained model
    # torch.save(model.state_dict(), 'model_trained')
    # print('Main Model Saved')
