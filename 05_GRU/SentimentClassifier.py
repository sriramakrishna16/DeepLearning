import torch
import torch.nn as nn
import torch.optim as optim

# creating small dataset

sentences = [
    "i love this movie",
    "this movie is great",
    "i like this film",
    "this film is amazing",

    "i hate this movie",
    "this movie is terrible",
    "i dislike this film",
    "this film is boring"
]

labels = torch.tensor([
    1,1,1,1,0,0,0,0
], dtype = torch.float32)


# creating vocabulary

vocab = {"<PAD>" : 0 , "<UNK>" :1}

for sentence in sentences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = len(vocab)

print(vocab)


sequences = []

for sentence in sentences:
    sequence = []

    for word in sentence.split(" "):
        sequence.append(vocab[word])

    sequences.append(sequence)

print(sequences)

max_length = max(len(seq) for seq in sequences)

padded_sequences = []

for seq in sequences:
    seq = seq + [vocab["<PAD>"]] * (max_length - len(seq))
    padded_sequences.append(seq)

print(padded_sequences)

X = torch.tensor(padded_sequences, dtype = torch.long)
print("input : \n", X)
print(X.shape)



# GRU model
class GRUSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx= 0
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        embedded = self.embedding(x)

        output, hidden = self.gru(embedded)

        final_hidden = hidden[-1]

        output = self.fc(final_hidden)

        return output

embedding_dim = 8
hidden_size = 10

model = GRUSentimentModel(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    hidden_size=hidden_size
)

print("\nModel:")
print(model)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 500

for epoch in range(epochs):

    # Forward pass
    outputs = model(X).squeeze(1)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        predictions = (torch.sigmoid(outputs) >= 0.5).float()

        accuracy = (predictions == labels).float().mean()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.4f} "
            f"Accuracy: {accuracy.item():.2f}"
        )


# 8. Test the model

def predict(sentence):

    words = sentence.lower().split()

    sequence = []

    for word in words:

        if word in vocab:
            sequence.append(vocab[word])
        else:
            sequence.append(vocab["<UNK>"])

    # Padding
    sequence = sequence + [0] * (max_length - len(sequence))

    x = torch.tensor([sequence], dtype=torch.long)

    with torch.no_grad():

        output = model(x)

        probability = torch.sigmoid(output).item()

    if probability >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    print("\nSentence:", sentence)
    print("Probability:", round(probability, 4))
    print("Prediction:", sentiment)


# prediction

predict("i love this movie")

predict("this movie is terrible")

predict("i like this film")

predict("this film is boring")