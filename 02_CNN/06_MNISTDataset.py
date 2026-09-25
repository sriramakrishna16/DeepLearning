import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using device :" , device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = datasets.MNIST(
    root = "./data",
    train = True,
    transform = transform,
    download = True
)

test_dataset = datasets.MNIST(
    root = "./data",
    train = False,
    transform = transform,
    download = True
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 64,
    shuffle = True
)

print("training samples ", len(train_dataset))
print("testing samples ", len(test_dataset))

# cnn model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 28 28
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = 3,
            padding = 1,
            stride = 1
        )
        # 16 28 28

        self.relu1 = nn.ReLU()

        # 16 14 14
        self.pool1 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        self.conv2 = nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        # 32 14 14

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )
        # 32 7 7

        self.fc1 = nn.Linear(
            32 * 7 * 7,
            128
        )

        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(
            128,
            10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.flatten(1)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)

        return x

model = CNN()

model = model.to(device)

criterian = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr = 0.001
)

epochs = 10

for epoch in range(epochs):
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images , labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterian(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _,predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(f"epoch [{epoch+1}/{epochs}]" f"loss : {epoch_loss:.4f}" f"accuracy : {epoch_accuracy:.2f}%")


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

print("\nTest Accuracy:",f"{test_accuracy:.2f}%")


# output

# epoch [1/10]loss : 0.1952 accuracy : 94.36%
# epoch [2/10]loss : 0.0536 accuracy : 98.38%
# epoch [3/10]loss : 0.0371 accuracy : 98.83%
# epoch [4/10]loss : 0.0283 accuracy : 99.14%
# epoch [5/10]loss : 0.0225 accuracy : 99.26%
# epoch [6/10]loss : 0.0173 accuracy : 99.45%
# epoch [7/10]loss : 0.0142 accuracy : 99.55%
# epoch [8/10]loss : 0.0115 accuracy : 99.65%
# epoch [9/10]loss : 0.0099 accuracy : 99.69%
# epoch [10/10]loss : 0.0090 accuracy : 99.68%

# Test Accuracy: 99.06%




