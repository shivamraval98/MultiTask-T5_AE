import pandas as pd
from sklearn.metrics import classification_report
from models.T5Model.t5_model import T5Model 
import os    
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

'''
Function to train the T5 model on a single task
Parameters
------------
task_type (str): The task type for training the model, choose from: assert_ade, ner_ade, ner_drug, ner_dosage
task_name (str): The task name to train the model, choose from: smm4h_task1, smm4h_task2, cadec, ade_corpus
model_name (str): The T5 pre-trained model name (t5-base, t5-small, t5-large)
model_args (dict): The training parameters used while training the T5 model
'''
def T5_train(task_type, task_name, model_name, model_args):
    train_df = pd.read_csv(os.getcwd() + "/data/combiner_data/" + task_type + "/train_" + task_type + "_" + task_name + ".csv")        
    eval_df = pd.read_csv(os.getcwd() + "/data/combiner_data/" + task_type + "/eval_" + task_type + "_" + task_name + ".csv")

    model = T5Model("t5", model_name, args = model_args)
    model.train_model(train_df, eval_data = eval_df)


def main():
    '''
    Model hyper-parameters, for detailed list of hyperparamets checkout model_args.py and global_args.py in /models/config
    The paramters description can also be found on https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    '''
    model_args = {
        "max_seq_length": 130,
        "train_batch_size": 80,
        "eval_batch_size": 8,
        "num_train_epochs": 12,
        "evaluate_during_training": True,
        "evaluate_during_training_steps": 500,
        "evaluate_during_training_verbose": True,
        "n_gpu": 4,
        "learning_rate": 1e-4,
        
        "evaluate_generated_text": True,
        "gradient_accumulation_steps": 1,
        
        "use_multiprocessing": False,
        "fp16": True,
        
        "save_steps": -1,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,    "reprocess_input_data": True,
        "overwrite_output_dir": True,
        
        "wandb_project": None
    }
    
    #Train the T5 model on the given task type and name
    T5_train("assert_ade", "smm4h_task1", "t5-base", model_args)
    
if __name__ == "__main__":
    main()