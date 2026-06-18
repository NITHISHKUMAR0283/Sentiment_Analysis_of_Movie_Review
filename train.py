import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from src.data_downloader import download_imdb
from src.data_reader import get_data
from src.PandasToTensor import PandasToTensor
from src.SentimentClassifier import SentimentClassifier
from sklearn.metrics import classification_report


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"device {device}")

download_imdb()

train,test = get_data()


train = PandasToTensor(train_data=train ,max_length=128)
test = PandasToTensor(train_data=test,max_length=128)

train_loader = DataLoader(train , batch_size = 64, shuffle = True)
test_loader = DataLoader(test , batch_size=64,shuffle = False)

model = SentimentClassifier(vocab_size=train.vocab_size)
model.to(device)

criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

scaler = torch.amp.GradScaler('cuda')

model.train()

num_epoch = 5
best_acc = 0

for epoch in range(num_epoch):
    avg_loss = 0
    total_sample = 0
    correct_pred = 0
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

        prediction = torch.argmax(logits,dim=1)
        correct_pred += (prediction == labels ).sum().item()
        total_sample += labels.size(0)

        avg_loss+=loss.item()
         
        accuracy = correct_pred/total_sample*100
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(),"./models/best_sentiment_model.pt")


    print(f"loss :{avg_loss/len(train_loader)}",f" accuracy :{accuracy}")

model.eval()

all_pred = []
all_labels = []
with torch.no_grad():
    avg_loss = 0
    corr_pred = 0
    total_sample = 0
    for batch in test_loader:
        input_ids=batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids=input_ids,attention_mask = attention_mask)

        loss = criterian(logits,labels)
        avg_loss +=loss.item()

        test_pred = torch.argmax(logits,dim=1)

        all_pred.extend(test_pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        corr_pred += (test_pred==labels).sum().item()
        total_sample+= labels.size(0)
    

    print(f"loss: {avg_loss/len(test_loader)} , accuracy : {corr_pred/total_sample*100})")

print(classification_report(all_labels,all_pred,target_names=["Negative","Positive"]))