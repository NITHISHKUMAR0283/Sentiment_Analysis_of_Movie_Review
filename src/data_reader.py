from datasets import Dataset
import pandas


def get_data ():
    train_data = Dataset.from_file(r"data\imdb\plain_text\0.0.0\e6281661ce1c48d982bc483cf8a173c1bbeb5d31\imdb-train.arrow")
    test_data  = Dataset.from_file(r"data\imdb\plain_text\0.0.0\e6281661ce1c48d982bc483cf8a173c1bbeb5d31\imdb-test.arrow")

    train_data = train_data.to_pandas()
    test_data = test_data.to_pandas()

    return train_data , test_data