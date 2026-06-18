import torch.nn as nn
import torch 

class SentimentClassifier(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = 128
        )
        self.fc1 = nn.Linear(128,2)
    
    def forward(self,input_ids,attention_mask):
        
        embedded = self.embeddings(input_ids)

        mask = attention_mask.unsqueeze(-1).float()

        masked_embedding = embedded * mask

        sum_embedding = torch.sum(masked_embedding,dim=1)
        actual_lengths = torch.clamp(mask.sum(dim=1),min=1)

        pooled = sum_embedding/actual_lengths

        output = self.fc1(pooled)

        return output