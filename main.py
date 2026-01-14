from dataset import DataSet
from train import Train

dataset = DataSet()
train_dataset, valid_dataset = dataset.get_dataset()
tokenized_train_dataset, tokenized_valid_dataset = dataset.get_tokenized_dataset()
data_collator = dataset.get_data_collator()

trainer = Train()
trainer.train_bert(train_dataset, valid_dataset, data_collator)