from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset
from utils import load_config

config = load_config("config.yaml")
set_seed(123)

class DataSet():
    def __init__(self):
        self.train_data_path = config["data"]["train_data_path"]
        self.max_length = config["model"]["h_param"]["max_length"]
        self.dataset = load_dataset("csv", data_files = {"train": self.train_data_path})
        self.dataset = self.dataset["train"].train_test_split(test_size = 0.2, seed = 123)
        self.train_dataset = self.dataset["train"]
        self.valid_dataset = self.dataset["test"]
        self.tokenizer_name = config["model"]["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.TEXT_COL = "full_text"
        self.LABEL_COL = "generated"
        
    def tokenize_fn(self, batch):
        return self.tokenizer(
            batch[self.TEXT_COL],
            truncation = True,
            max_length = self.max_length,
        )
        
    def get_dataset(self):
        return self.train_dataset, self.valid_dataset
    
    def get_tokenized_dataset(self):
        train_dataset = self.train_dataset.map(self.tokenize_fn, batched = True)
        valid_dataset = self.valid_dataset.map(self.tokenize_fn, batched = True)
        return train_dataset, valid_dataset
    
    def get_data_collator(self):
        data_collator = DataCollatorWithPadding(tokenizer = self.tokenizer)
        return data_collator
