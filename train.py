import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch.nn as nn
from torch.utils.data import random_split


"""
This is part of some pre-processing, may be require more but for now we have our train
loader, that contains images that we need to train in total we have 2876 images, so train loader
picks up those images and label the class as:
 Mapping of numeric labels to class name - {0: 'angry', 1: 'boredom', 2: 'engaged', 3: 'neutral'}
and batch size is 32, so in total 90 batches.
"""
# Neural networks work with tensors
# Define transformations for data preprocessing

def dataLoading():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load data from folders
    train_data = datasets.ImageFolder('train/', transform=transform)
    validation_data = datasets.ImageFolder('validation/', transform=transform)
    test_data = datasets.ImageFolder('test/', transform=transform)


    # train_data, val_data = random_split(train_data,[train_size,val_size])

    # Create data loaders
    #pytorch - channel, height, width
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # className = train_loader.dataset.classes
    # #print(f'classname- {className}')
    #
    # labels_to_class = {i : className[i] for i in range(len(className))}
    # #print(f'Mapping of numeric labels to class name - {labels_to_class}')
    #
    # total_batches = len(train_loader)
    #
    # #print(f'Total batches {total_batches}')
    #
    # data_iter = iter(train_loader)
    #
    # total_images = 0
    #
    # for i in range(total_batches):
    #     images, labels = next(data_iter)
    #     class_count = defaultdict(int)
    #
    #     for label in labels.tolist():
    #         total_images += 1
    #         class_count[label] += 1
    #
    #     print(f'The batch {i+1}: Image per class  - {dict(class_count)}')
    #
    # print(f'Total Images done by computation {total_images}')
    # print(f'Total Images done by train_loader {len(train_loader.dataset)}')

    return train_loader , val_loader, test_loader

    # for i in range(4):
    #     images, labels = next(data_iter)
    #
    # image_to_show = 3
    # #matplot - height, width, channel
    # image = np.transpose(images[image_to_show],(1,2,0))
    # label = labels[image_to_show]
    #
    # plt.imshow(image)
    # plt.title(f'class: {label.item()}')
    # plt.show()



"""
This is part of training the model and see its accuracy. Right now I'm training on full dataset so no 
test-dataset or validation set, and I dont what will be architecture, lets see we need how many layers
or pooling or which type of loss function and what will be output of feature-map that comes out of these layers
"""
"""
Key terms :

nn.Module, which is the base class for all neural network modules in PyTorch.
In the case of a PyTorch model, calling super().__init__() in the constructor ensures that 
the base class (nn.Module in this case) is properly initialized, setting up necessary structures and 
functionalities that are crucial for neural network implementations in PyTorch


nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1) -
 
3 represents the input channels (assuming RGB images).
16 is the number of output channels or filters.
kernel_size=3 specifies a 3x3 convolutional kernel.
stride=1 defines the step size of the kernel during the convolution.
padding=1 adds a 1-pixel border around the input to maintain spatial dimensions after convolution.

"""
class CNNModel(nn.Module):

    def __init__(self,num_of_classes):
        super(CNNModel,self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(4,4)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)
        self.fc2 = nn.Linear(10, num_of_classes)

    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))#256->256/4 = 64
        x = self.pool(torch.relu(self.conv2(x)))#64/4 = 16
        x = x.view(-1,32*16*16)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient computation during validation
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(val_loader)

    return accuracy, avg_val_loss


train_loader,val_loader, test_loader = dataLoading()

model = CNNModel(4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(running_loss/len(train_loader))

accuracy, avg_val_loss = evaluate_model(model, val_loader, criterion)
print(f'Validation Accuracy: {accuracy:.2f}%')
print(f'Average Validation Loss: {avg_val_loss:.4f}')

accuracy,avg_test_loss = evaluate_model(model,test_loader,criterion)
print(f'Test Accuracy: {accuracy:.2f}%')
print(f'Test Validation Loss: {avg_val_loss:.4f}')

torch.save(model.state_dict(), 'model1')








