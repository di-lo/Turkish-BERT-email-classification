import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# progress bars
from tqdm import tqdm

df = pd.read_csv("haberler.csv")

# Encode labels to integers
label_encoder = LabelEncoder()
df['etiket'] = label_encoder.fit_transform(df['etiket'])

# 3. Split into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['haber'].tolist(), df['haber'].tolist(), test_size=0.2, random_state=42, stratify=df['etiket'])

# 4. Tokenizer and encoding
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['etiket'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_data, train_labels)
test_dataset = NewsDataset(test_data, test_labels)

# 5. Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 6. Training setup
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 7. Training loop
model.train()
for epoch in range(3):  # try more epochs for better results !!!!!!!!
    print(f"Epoch {epoch+1}")
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 8. Evaluation
model.eval()
test_loader = DataLoader(test_dataset, batch_size=8)
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

# 9. Classification report
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))


# loss function
# check accuracy
    # check_accuracy(val_loader, model, device=DEVICE)
# dice score