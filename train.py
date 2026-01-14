from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from utils import load_config
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np

config = load_config("config.yaml")

class Train():
    def __init__(self):
        self.training_args = TrainingArguments(
            output_dir = config["train"]["output_dir"],
            eval_strategy = config["train"]["eval_strategy"],
            save_strategy = config["train"]["eval_strategy"],
            logging_strategy = config["train"]["logging_strategy"],
            logging_steps = config["train"]["logging_steps"],

            per_device_train_batch_size = config["train"]["per_device_train_batch_size"],
            per_device_eval_batch_size = config["train"]["per_device_eval_batch_size"],
            learning_rate = config["train"]["lr"],
            num_train_epochs = config["train"]["epochs"],
            weight_decay = config["train"]["weight_decay"],
            warmup_ratio = config["train"]["warmup_ratio"],

            fp16 = torch.cuda.is_available(),
            load_best_model_at_end = True,
            metric_for_best_model = config["train"]["metric_for_best_model"],
            greater_is_better = True,

            save_total_limit = 2,
            report_to = "none",
        )
        
        self.model_name = config["model"]["model_name"]
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels = 2,
        )
        
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis = -1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds)
        }
    
    def train_bert(self, train_dataset, valid_dataset, data_collator):
        trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = train_dataset,
            eval_dataset = valid_dataset,
            data_collator = data_collator,
            compute_metrics = self.compute_metrics,
        )
        trainer.train()
        return
    
