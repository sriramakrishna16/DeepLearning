import kagglehub

# Download latest version
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
                                  output_dir = "./data")

print(path)

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/IMDB Dataset.csv")
print(df.shape)
print(df.head(10))

texts = df["review"].values #converts to numpy array
labels = df["sentiment"].map({
    "positive": 1,
    "negative": 0
}).values

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>','', text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
texts = [clean_text(t) for t in texts]


def build_vocab(texts, max_vocab = 20000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {"<PAD>":0 , "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab):
        vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(texts)

vocab_size = len(vocab)

def encode(text):
    return [vocab.get(w, vocab["<UNK>"]) for w in text.split()]
encoded = [encode(t) for t in texts]

max_len = 200

def pad(seq):
    if len(seq) < max_len:
        return seq + [0] * (max_len - len(seq))
    return seq[:max_len]
padded = [pad(seq) for seq in encoded]

x_train, x_test, y_train, y_test = train_test_split(
    padded, labels , test_size = 0.2, random_state=42
)

class IMDBDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_loader = DataLoader(IMDBDataset(x_train, y_train), batch_size=64, shuffle= True)
test_loader = DataLoader(IMDBDataset(x_test, y_test), batch_size = 64)

class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim = 128, hidden_dim = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        out = output[:, -1, :]
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleRNNModel(vocab_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(x_batch)
        loss = criterion(torch.sigmoid(outputs), y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch : ", epoch+1 , "Loss : ", total_loss)





