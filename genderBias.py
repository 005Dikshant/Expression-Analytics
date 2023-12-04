import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from main import CNNModel
import matplotlib.pyplot as plt

def data_loading(data_folder):
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

def get_predictions(model, loader, device):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradient computation during validation
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def showAnalysis(y_test, y_pred):

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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    female_data_loader = data_loading(data_folder='gender/female')
    male_data_loader = data_loading(data_folder='gender/male')

    print('Male and Female Class Images Loaded:')

    model = CNNModel(4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Load trained model
    model.load_state_dict(torch.load('model_trained4', map_location=device))

    y_pred, y_test = get_predictions(model,female_data_loader,device)
    acc = accuracy_score(y_test, y_pred)

    showAnalysis(y_test, y_pred)

    y_pred, y_test = get_predictions(model,male_data_loader,device)
    acc = accuracy_score(y_test, y_pred)
    showAnalysis(y_test, y_pred)


# senior_iterator = iter(senior_data_loader)
# senior_images, labels = next(senior_iterator)
# senior_image = senior_images[0].permute(1, 2, 0).numpy()  # Change tensor shape for visualization
# plt.figure()
# plt.imshow(senior_image)
# plt.title(f'Senior - First Image')
# plt.show()
#
# # Display the first image from the young_data_loader
# young_iterator = iter(young_data_loader)
# young_images, test_labels = next(young_iterator)
# young_image = young_images[0].permute(1, 2, 0).numpy()  # Change tensor shape for visualization
# plt.figure()
# plt.imshow(young_image)
# plt.title(f'Young - First Image')
# plt.show()


# unique_labels = set()
# for _, labels in senior_data_loader:
#     unique_labels.update(labels.numpy())
#
# num_classes_in_test_loader = len(unique_labels)
# print(f'Number of classes in senior_data_loader: {num_classes_in_test_loader}')
#
# class_counts = {label: 0 for label in range(4)}  # Assuming you have 4 classes
# for _, labels in senior_data_loader:
#     for label in labels.numpy():
#         class_counts[label] += 1
#
# print('Class quantities in senior_data_loader:')
# for label, count in class_counts.items():
#     print(f'Class {label}: {count} samples')
#
#
# unique_labels = set()
# for _, labels in young_data_loader:
#     unique_labels.update(labels.numpy())
#
# num_classes_in_test_loader = len(unique_labels)
# print(f'Number of classes in young_data_loader: {num_classes_in_test_loader}')
#
# class_counts = {label: 0 for label in range(4)}  # Assuming you have 4 classes
# for _, labels in young_data_loader:
#     for label in labels.numpy():
#         class_counts[label] += 1
#
# print('Class quantities in young_data_loader:')
# for label, count in class_counts.items():
#     print(f'Class {label}: {count} samples')