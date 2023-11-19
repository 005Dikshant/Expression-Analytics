import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, \
    f1_score
import matplotlib.pyplot as plt
from variant2 import CNNModel


def data_loading(data_folder='new_dataset/', batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data = datasets.ImageFolder(data_folder, transform=transform)

    train_ratio, val_ratio = 0.7, 0.15
    total_samples = len(data)

    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)

    train_data, temp_data = train_test_split(data, train_size=train_samples,
                                             random_state=42)
    test_data, validation_data = train_test_split(temp_data,
                                                  test_size=val_samples,
                                                  random_state=42)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_data,
                                             batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader


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


def visualize_predictions(model, loader, classes, device, num_samples=5):
    model.eval()
    samples = iter(loader)

    for _ in range(num_samples):
        inputs, labels = next(samples)
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

        plt.imshow(transforms.ToPILImage()(inputs[0].cpu()))
        plt.title(
            f"Actual: {classes[labels[0].item()]}, Predicted: {classes[predicted[0].item()]}")
        plt.show()


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, val_loader, test_loader = data_loading()
print(f'Data Loaded: Test Images {len(test_loader.dataset)}')

# Initialize model
model = CNNModel(4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Load trained model
model.load_state_dict(torch.load('model_trained3', map_location=device))

# Evaluate model
accuracy, avg_test_loss, cm, y_pred, y_test = evaluate_model(model,
                                                             test_loader,
                                                             criterion, device)
print(f'Test Accuracy: {accuracy:.2f}%')
print(f'Average Test Loss: {avg_test_loss:.4f}')

# Visualize predictions
visualize_predictions(model, test_loader,
                      classes=['Angry', 'Boredom', 'Engaged', 'Neutral'],
                      device=device, num_samples=5)

# Plot confusion matrix with blue color
classes = ['Angry', 'Boredom', 'Engaged', 'Neutral']
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cm_display.plot(cmap='Blues')
plt.savefig('confusion_matrix3.png')
plt.show()

# Calculate additional metrics
acc = accuracy_score(y_test, y_pred)
pre_macro = precision_score(y_test, y_pred, average='macro')
pre_micro = precision_score(y_test, y_pred, average='micro')
recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

# Print additional metrics
print(f'Test Accuracy from SkLearn: {acc * 100:.2f}%')
print(f'Test Precision (Macro): {pre_macro * 100:.2f}%')
print(f'Test Precision (Micro): {pre_micro * 100:.2f}%')
print(f'Test Recall (Macro): {recall_macro * 100:.2f}%')
print(f'Test Recall (Micro): {recall_micro * 100:.2f}%')
print(f'Test F1 Score (Macro): {f1_macro * 100:.2f}%')
print(f'Test F1 Score (Micro): {f1_micro * 100:.2f}%')

