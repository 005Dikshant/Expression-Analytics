import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from main import CNNModel

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

def evaluate_model(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradient computation during validation
        for inputs, labels in loader:
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

    return accuracy, avg_loss,cm,all_preds,all_labels

train_loader, val_loader, test_loader = dataLoading()
print(f'Data Loaded: Test Images {len(test_loader)*32}')

model = CNNModel(4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.load_state_dict(torch.load('model_trained'))

accuracy, avg_test_loss,cm,y_pred,y_test = evaluate_model(model, test_loader, criterion)
print(f'Test Accuracy: {accuracy:.2f}%')
print(f'Average Test Loss: {avg_test_loss:.4f}')

classes = ['Angry', 'Boredom', 'Engaged', 'Neutral']
ConfusionMatrixDisplay(cm, display_labels=classes).plot()
plt.savefig('confusion_matrix.png')
plt.show()

acc = accuracy_score(y_test, y_pred)
pre_macro = precision_score(y_test, y_pred, average='macro')
pre_micro = precision_score(y_test, y_pred, average='micro')
recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')

print(f'Test Accuracy from SkLearn: {acc:.2f}%')
print(f'Test pre_macro from SkLearn: {pre_macro:.2f}%')
print(f'Test pre_micro from SkLearn: {pre_micro:.2f}%')
print(f'Test recall_macro from SkLearn: {recall_macro:.2f}%')
print(f'Test recall_micro from SkLearn: {recall_micro:.2f}%')
