import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch 
from models.bert_dataset_loader import *

'''
PyTorch Dataset Class for the dataset iterator
'''
class SMM4HDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        enc_item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item = {}
        item['input_ids'] = enc_item['input_ids']
        item['labels'] = torch.tensor(self.labels[idx])
        
        return item
    
    def __len__(self):
        return len(self.labels)

'''
BERT Model Class
Parameters
-----------
model_name (str): the pre-trained model name from the huggingface transformers library
task (str): the task name (ex. smm4h_task1, cadec ..)
'''
class BaselineBERT(object):
    def __init__(self, model_name, task):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.data_dict = Dataset_obj(task).get_data_dict()

    '''
    Function to prepare the tokenized encodings and dataset class for each split
    '''
    def get_torch_dataset(self):
        max_len = 130
        train_encodings = self.tokenizer(self.data_dict['train_text'], truncation = True, padding = True, max_length = max_len)
        eval_encodings = self.tokenizer(self.data_dict['eval_text'], truncation = True, padding = True, max_length = max_len)
        test_encodings = self.tokenizer(self.data_dict['test_text'], truncation = True, padding = True, max_length = max_len)
        
        train_dataset = SMM4HDataset(train_encodings, self.data_dict["train_labels"])
        eval_dataset = SMM4HDataset(eval_encodings, self.data_dict["eval_labels"])
        test_dataset = SMM4HDataset(test_encodings, self.data_dict["test_labels"])
        
        return train_dataset, eval_dataset, test_dataset
    
