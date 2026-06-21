import torch.nn as nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,2
        )
            
    def forward(self,input_ids,attention_mask):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_embeddings = outputs.last_hidden_state[:,0,:]

        cls_embeddings = self.dropout(cls_embeddings)

        logits = self.classifier(cls_embeddings)

        return logits