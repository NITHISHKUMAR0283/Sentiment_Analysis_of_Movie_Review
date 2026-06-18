import torch
from torch.utils.data import Dataset,DataLoader
from src.data_downloader import download_imdb
from src.data_reader import get_data
from src.PandasToTensor import PandasToTensor

download_imdb()

train,test = get_data()


train = PandasToTensor(train_data=train ,max_length=128)

train_loader = DataLoader(train , batch_size = 32, shuffle = True)

print(next(iter(train_loader)))