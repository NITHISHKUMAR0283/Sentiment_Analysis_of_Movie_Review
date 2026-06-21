import torch
from torch.utils.data import Dataset,DataLoader,random_split
import torch.nn as nn
from src.data_downloader import download_imdb
from src.data_reader import get_data
from src.PandasToTensor import PandasToTensor
from src.SentimentClassifier import SentimentClassifier
from sklearn.metrics import classification_report



device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"device {device}")

download_imdb()

train_raw, test_raw = get_data()


full_train_dataset = PandasToTensor(train_data=train_raw, max_length=128)
test_dataset = PandasToTensor(train_data=test_raw, max_length=128)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42) # Fixed seed for reproducible splits
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = SentimentClassifier(vocab_size=full_train_dataset.vocab_size)
model.to(device)

criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))



num_epoch = 10
history = {"train_loss": [], "val_loss": [], "val_acc": []}


for epoch in range(num_epoch):
    
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast('cuda'):
            logits = model(input_ids=input_ids,attention_mask = attention_mask )
            loss = criterian(logits,labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)


    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids=batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterian(logits, labels)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    avg_val_loss = val_loss / len(val_loader)
    val_acc = (correct / total) * 100

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch [{epoch+1:02d}/{num_epoch:02d}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")


print("\nRunning Final Evaluation on Test Dataset...")
model.eval()
all_pred = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
        test_pred = torch.argmax(logits, dim=1)
        all_pred.extend(test_pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nFinal Classification Report (Test Data):")
print(classification_report(all_labels, all_pred, target_names=["Negative", "Positive"]))