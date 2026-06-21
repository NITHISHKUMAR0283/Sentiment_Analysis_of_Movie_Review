import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch 

class SentimentClassifier(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = 128
        )
        self.lstm = nn.LSTM(
            input_size = 128,
            hidden_size = 128,
            batch_first= True,
            bidirectional = True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256,2)
    
    def forward(self,input_ids,attention_mask):
        
        max_len = input_ids.size(1)

        embedded = self.embeddings(input_ids)
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()

        packed_embedded = pack_padded_sequence(
            embedded, lengths , batch_first = True,enforce_sorted=False
        )

        packed_output , _ = self.lstm(packed_embedded)
        lstm_output ,_= pad_packed_sequence(packed_output,batch_first = True,total_length=max_len)
        
        key_padding_mask = (attention_mask==0)

        attn_output,_ = self.attention(
            query=lstm_output,
            key = lstm_output,
            value = lstm_output,
            key_padding_mask = key_padding_mask
        )
        mask = attention_mask.unsqueeze(-1).float()

        masked_embedding = attn_output * mask

        sum_embedding = torch.sum(masked_embedding,dim=1)
        actual_lengths = torch.clamp(mask.sum(dim=1),min=1)

        pooled = sum_embedding/actual_lengths
        pooled = self.dropout(pooled)

        output = self.fc1(pooled)

        

        return output