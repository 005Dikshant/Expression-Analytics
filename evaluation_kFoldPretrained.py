import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from main import CNNModel
import matplotlib.pyplot as plt

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

    cm = confusion_matrix(all_labels, all_preds)

    return all_preds, all_labels, cm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNNModel(4).to(device)

# Load trained model
model.load_state_dict(torch.load('model_trained', map_location=device))

data_loader = data_loading()
print('Running Evaluation on Pre-trained model')
print('Dataset loaded for evaluation')
full_dataset_size = len(data_loader.dataset)

k_folds = KFold(n_splits=10,shuffle=True,random_state=42)

for fold, (_,test_index) in enumerate(k_folds.split(range(full_dataset_size))):

    test_loader = DataLoader(torch.utils.data.Subset(data_loader.dataset, test_index), shuffle=True,batch_size=32)

    # unique_labels = set()
    # for _, labels in test_loader:
    #     unique_labels.update(labels.numpy())
    #
    # num_classes_in_test_loader = len(unique_labels)
    # print(f'Number of classes in test_loader: {num_classes_in_test_loader}')
    #
    # class_counts = {label: 0 for label in range(4)}  # Assuming you have 4 classes
    # for _, labels in test_loader:
    #     for label in labels.numpy():
    #         class_counts[label] += 1
    #
    # print('Class quantities in test_loader:')
    # for label, count in class_counts.items():
    #     print(f'Class {label}: {count} samples')


    y_pred, y_test, cm = get_predictions(model,test_loader,device)

    # classes = ['Angry', 'Boredom', 'Engaged', 'Neutral']
    # cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # cm_display.plot(cmap='Blues')
    # plt.show()

    print(f'After Fold: {fold + 1}')
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



# train_iterator = iter(train_loader)
# train_images, train_labels = next(train_iterator)
# train_image = train_images[0].permute(1, 2, 0).numpy()  # Change tensor shape for visualization
# plt.imshow(train_image)
# plt.title(f'Fold {fold + 1}: Train Loader - First Image')
# plt.show()
#
# # Display the first image from the test loader
# test_iterator = iter(test_loader)
# test_images, test_labels = next(test_iterator)
# test_image = test_images[0].permute(1, 2, 0).numpy()  # Change tensor shape for visualization
# plt.imshow(test_image)
# plt.title(f'Fold {fold + 1}: Test Loader - First Image')
# plt.show()