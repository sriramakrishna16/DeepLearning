from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()

x = data.data
y = data.target

print(data.data.shape)
print(data.feature_names)
print(data.target_names)

# print(x[:5])
# print(y[:5])

# print(np.isnan(x).sum())
# print(np.isnan(y).sum())

# print(x.min(axis=0))
# print(x.max(axis=0))

for i, name in enumerate(data.feature_names):
    print(i, name)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)


from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

# print(x_train.min(axis=0))
# print(x_test.max(axis=0))

import torch

x_train = torch.tensor(x_train, dtype= torch.float32)
x_test = torch.tensor(x_test, dtype= torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(30, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = ANN()

output = model(x_train)
print(output.shape)
# print(output[:5])

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)

for epoch in range(200):
    output = model(x_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{100}], "
            f"Loss: {loss.item():.4f}"
        )


model.eval()

with torch.no_grad():
    predictions = model(x_test)

predicted_classes = (predictions >= 0.5).float()

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(
    y_test.numpy(),
    predicted_classes.numpy()
)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test.numpy(),
        predicted_classes.numpy()
    )
)


# by using batchnorm and Adam optimizer with learning rate = 0.01 got 97.36% accuracy