import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\Visom6\Vison6 Text Classifier\nlp_dataset.csv")

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Split data into training and testing sets
text_train, text_test, target_train, target_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encoding text
train_encodings = tokenizer(list(text_train), truncation=True, padding=True)
test_encodings = tokenizer(list(text_test), truncation=True, padding=True)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, target_train)
test_dataset = TextDataset(test_encodings, target_test)

# DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

# Initialize BERT model
num_labels = 4
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(model.device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation loop
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(model.device)
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / len(loader)

# Training
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer)
    val_loss = evaluate(model, test_loader)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Classification function
def classify_text(new_text, model, tokenizer):
    inputs = tokenizer(new_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return prediction

# Classify new texts
new_text1 = "Our agency excels in delivering real business value through data science."
prediction1 = classify_text(new_text1, model, tokenizer)
print(new_text1)
print("Prediction for new_text1:", prediction1)

# Add more texts if needed
