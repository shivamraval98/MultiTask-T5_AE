from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch 
import torch.nn as nn
from sklearn.metrics import classification_report
import logging
from models.bert_dataset_loader import *
from models.bert_model import *

'''
Trainer for the BERT model
Parameters
-----------
class_weight (torch.Tensor): the binary class label weights for weighted cross entropy loss
'''
class MyTrainer(Trainer):
    def init_weights(self,class_weight):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_weight = class_weight.to(device)
        
    '''
    Custom Loss function of weighted cross entropy loss
    '''
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weight)
        loss = loss_fn(logits, labels)
        return loss
 
'''
Function to calculate the metrics for the predictionsParameters
'''    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print("Validation Report")
    report = classification_report(labels, preds)
    print(report)
    return {
        'report': report
    }

    
'''
Function to train the BERT model
Parameters
------------
model_name (str): pre-trained model name from the huggingface transformers library
task_name (str): task name (ex. smm4h_task1, cadec ..)
training_args (dict): the training arguments to be used for training the BERT model
'''
def run_model(model_name, task_name, training_args):
    model_obj = BaselineBERT(model_name, task_name)
    train_dataset, eval_dataset, _ = model_obj.get_torch_dataset()
    

    trainer = MyTrainer(
        model=model_obj.model,                        
        args=training_args, 
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,   
        eval_dataset=eval_dataset,
    )
    trainer.init_weights(model_obj.data_dict['class_weights'])

    logging.basicConfig(level = logging.INFO)

    trainer.train()
    sub_name = "baseline"
    trainer.save_model('./' + sub_name)
    

def main():
    #choose any pre-trained BERT based models from huggingface's transformers library (ex: dmis-lab/biobert-v1.1, allenai/scibert_scivocab_uncased, ...)
    model_name = "emilyalsentzer/Bio_ClinicalBERT"  
    #choose from the following assert_ade task to train the model (smm4h_task1, smm4h_task2, cadec, ade_corpus)
    task_name = "smm4h_task1"

    
    training_args = TrainingArguments(
        output_dir = './results',  #Output directory to store the model
        save_total_limit = 1,  #Number of checkpoints stored in the directory
        num_train_epochs=5,   #Number of training epochs
        per_device_train_batch_size=40,   #Training batch size per GPU
        per_device_eval_batch_size=64,   #Evaluation batch size per GPU
        warmup_steps=500,     #Warmup steps for the linear shcedule warmup used in Adam Optimizer
        weight_decay=0.01,    #Weight decay for the parameters in the models           
        logging_dir='./logs', #Path to the logs file
    )
    
    #Trains the model for the given task and arguments
    run_model(model_name, task_name, training_args)
    
    
    
if __name__ == "__main__":
    main()
