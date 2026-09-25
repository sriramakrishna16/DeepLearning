# this dataset contain 60,000 images with 10 classes
# data is divided into 50,000 images for training and 10000 for testing

# we are training cnn model using this dataset

import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root = "./data",
    train = True,
    download = True,
    transform = transform
)

test_dataset = datasets.CIFAR10(
    root = "./data",
    train = False,
    download = True,
    transform = transform
)

print("Training images : ", len(train_dataset))
print("Testing images : ", len(test_dataset))

image , label = train_dataset[100]

print("image shape : ", image.shape)
print("Label : ", label)

# displaying one image

# import matplotlib.pyplot as plt
# image , label = train_dataset[0]

# # matplot expects image in the form of 32, 32, 3 but tensor has 3,32,32
# image = image.permute(1,2,0)

# plt.figure(figsize=(4,4))
# plt.imshow(image)
# plt.title(train_dataset.classes[label])
# plt.axis("off")
# plt.show()

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

images, labels = next(iter(train_loader))

# print("Images shape:", images.shape)
# print("Labels shape:", labels.shape)

import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 5))

# for i in range(8):
#     image = images[i].permute(1, 2, 0)

#     plt.subplot(2, 4, i + 1)
#     plt.imshow(image)
#     plt.title(train_dataset.classes[labels[i]])
#     plt.axis("off")

# plt.tight_layout()
# plt.show()

import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device ", device)


# CNN model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 16,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        self.bn1 = nn.BatchNorm2d(16)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        ) 

        # second cnn block

        self.conv2 = nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3,
            padding = 1,
            stride = 1
        )

        self.bn2 = nn.BatchNorm2d(32)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        self.fc1 = nn.Linear(
            32 * 8 * 8,
            128
        )

        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(
            128,
            10
        )

    def forward(self, x):

        # batch, 3 , 32, 32
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # batch , 16, 16 , 16
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # batch, 32, 8, 8

        # now flatten
        x = x.flatten(1)


        # batch , 2048
        x = self.fc1(x)
        x = self.relu3(x)

        # x = self.dropout(x)


        # batch , 128
        x = self.fc2(x)

        # batch , 10

        return x

model = CNN()

model = model.to(device)

print("model ", model)

criterion = nn.CrossEntropyLoss()

# optimizer

import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr = 0.001
)

# training

epochs = 10

for epoch in range(epochs):
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# testing 

model.eval()

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


test_accuracy = 100 * correct / total

print("\nTest Accuracy:", f"{test_accuracy:.2f}%")


