import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# 01 type 1 feature extraction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = True,
    download = True,
    transform = transform
)

testset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = False,
    download = True,
    transform =  transform
)

from torch.utils.data import DataLoader

trainloader = DataLoader(
    trainset,
    batch_size = 64,
    shuffle = True
)

model = models.vgg16(weights = models.VGG16_Weights.DEFAULT)

# we call convolution base as features and fc layer as classifier part

# making traing false or freeze
for param in model.features.parameters():
    param.requires_grad = False

# vgg16 has total 3 fc layers fc1 , relu , dropout , fc2 , relu , dropout , fc3
# index 6 is the final layer

# this is architectural change
model.classifier[6] = nn.Linear(4096, 10) 

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.classifier.parameters(),
    lr = 0.001
)

for epoch in range(5):
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print("Epoch", epoch+1, "Loss : ", loss.item())


# 02 type 2 fine tuning
""" 
model1 = models.vgg16(Weights = models.VGG16_Weights.DEFAULT)
# the vgg16 has 138.36 million parameters

for param in model1.features.parameters():
    param.requires_grad = False

for param in model1.features[24:].parameters():
    param.requires_grad = True

model1.classifier[6] = nn.Linear(4096, 10)

model1 = model1.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model1.parameters(), lr = 0.001
)

for epoch in range(5):
    for images , labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model1(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("Epoch ", epoch+1, "Loss : ", loss.item()) """