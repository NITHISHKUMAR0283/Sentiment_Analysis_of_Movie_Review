import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
class PandasToTensor(Dataset):
    def __init__(self,train_data,max_length):

        self.train_text = train_data['text'].tolist()
        self.label = torch.tensor(train_data['label'].tolist(),dtype=torch.long)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size

    def __getitem__(self, index):
        token = self.tokenizer(self.train_text[index],
                               max_length = self.max_length,
                               padding  = "max_length",
                               truncation = True,
                               return_tensors = "pt"
                               )
        return {
        "input_ids": token["input_ids"].squeeze(0),
        "attention_mask": token["attention_mask"].squeeze(0),
        "labels": self.label[index]
    }

    def __len__(self):
            return len(self.train_text) 