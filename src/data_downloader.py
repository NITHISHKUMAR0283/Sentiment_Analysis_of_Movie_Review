import os 
from datasets import load_dataset

def download_imdb():

    path = r"C:\Users\USER\Desktop\AI\Projects\Sentiment_analysis\data"
    imdb_dataset = load_dataset("imdb",cache_dir = path)
    print('data downloaded')
