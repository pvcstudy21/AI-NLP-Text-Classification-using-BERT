import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\Visom6\Vison6 Text Classifier\nlp_dataset.csv")


# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


# Apply preprocessing to text data
df["text"] = df["text"].apply(preprocess_text)

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# Split data into training and testing sets
text_train, text_test, target_train, target_test = train_test_split(df["text"], df["label"], test_size=0.2,
                                                                    random_state=42)

# Tokenization and Vocabulary
tokenizer = nltk.word_tokenize
vocab = set()

for text in text_train:
    vocab.update(tokenizer(text))

vocab = {word: idx + 1 for idx, word in enumerate(vocab)}  # Reserve 0 for padding
vocab_size = len(vocab) + 1  # +1 for padding


# Encode text sequences
def encode_text(text, vocab):
    return [vocab[word] for word in tokenizer(text) if word in vocab]


text_train_enc = [encode_text(text, vocab) for text in text_train]
text_test_enc = [encode_text(text, vocab) for text in text_test]

# Pad sequences
text_train_enc = pad_sequence([torch.tensor(seq) for seq in text_train_enc], batch_first=True)
text_test_enc = pad_sequence([torch.tensor(seq) for seq in text_test_enc], batch_first=True)

train_labels = torch.tensor(target_train.tolist(), dtype=torch.long)  # Ensure dtype is long
test_labels = torch.tensor(target_test.tolist(), dtype=torch.long)  # Ensure dtype is long


# Create Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


train_dataset = TextDataset(text_train_enc, train_labels)
test_dataset = TextDataset(text_test_enc, test_labels)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        out = self.fc(lstm_out)
        return out


# Model parameters
embedding_dim = 128
hidden_dim = 256
output_dim = len(le.classes_)

# Initialize model, loss function, and optimizer
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training function
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for texts, labels in loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for texts, labels in loader:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)


# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss = evaluate(model, test_loader, criterion)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# Function to classify new text
def classify_text(new_text, model, vocab):
    new_text = preprocess_text(new_text)
    encoded_text = encode_text(new_text, vocab)
    padded_text = torch.tensor(encoded_text).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(padded_text)
        prediction = torch.argmax(outputs, dim=-1).item()
    return prediction


# Map numeric labels to original class labels
label_map = {index: label for index, label in enumerate(le.classes_)}

# Classify new texts
new_text1 = "Our agency excels in delivering real business value through data science"
prediction1 = classify_text(new_text1, model, vocab)
print(new_text1)
print("Prediction for new_text1:", label_map[prediction1])

new_text2 = "Looking for advice on how to stay motivated while learning data science."
prediction2 = classify_text(new_text2, model, vocab)
print(new_text2)
print("Prediction for new_text2:", label_map[prediction2])

new_text3 = "Developed a model for customer churn prediction. The results have been very insightful!"
prediction3 = classify_text(new_text3, model, vocab)
print(new_text3)
print("Prediction for new_text3:", label_map[prediction3])

new_text4 = "our highly trained staff are so great they can do anything, they are best in the business"
prediction4 = classify_text(new_text4, model, vocab)
print(new_text4)
print("Prediction for new_text4:", label_map[prediction4])

new_text5 = "i want to gain knowledge about data science, what is the best place to go"
prediction5 = classify_text(new_text5, model, vocab)
print(new_text5)
print("Prediction for new_text5:", label_map[prediction5])