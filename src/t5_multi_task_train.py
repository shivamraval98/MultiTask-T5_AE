import pandas as pd
import os
from models.T5MultiTask.t5_model import T5Model_MultiTask
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

'''
Function to train the T5 model in a multi-task and dataset scenario
Parameters
------------
train_task_dict (dict): dataset dictonary containing every datasets avaliable for each training task
model_name (str): The T5 pre-trained model name (t5-base, t5-small, t5-large)
model_args (dict): The training parameters used while training the T5 model
eval_task_dict (dict): dataset dictonary containing every datasets avaliable for each evaluation task

'''
def T5_Multi_Task_Train(train_task_dict, model_name, model_args, eval_task_dict=None):
    model = T5Model_MultiTask("t5", model_name, args = model_args)
    model.train_model(train_task_dict, eval_data = eval_task_dict)
    
    
def main():
    #dataset dictonary containing every datasets avaliable for each training task
    train_task_dict = {"assert_ade": ["assert_ade/train_assert_ade_smm4h_task1.csv", 
                                      "assert_ade/train_assert_ade_smm4h_task2.csv",
                                      "assert_ade/train_assert_ade_cadec.csv",
                                      "assert_ade/train_assert_ade_ade_corpus.csv"],
                       "ner_ade": ["ner_ade/train_ner_ade_smm4h_task2.csv",
                                   "ner_ade/train_ner_ade_cadec.csv",
                                   "ner_ade/train_ner_ade_ade_corpus.csv"],
                       "ner_drug": ["ner_drug/train_ner_drug_smm4h_task2.csv",
                                   "ner_drug/train_ner_drug_cadec.csv",
                                   "ner_drug/train_ner_drug_ade_corpus.csv"],
                       "ner_dosgae": ["ner_dosage/train_ner_dosage_ade_corpus.csv"]
                       
                       }
    '''
    eval_task_dict = {"assert_ade": ["assert_ade/eval_assert_ade_smm4h_task1.csv", 
                                      "assert_ade/eval_assert_ade_smm4h_task2.csv",
                                      "assert_ade/eval_assert_ade_cadec.csv",
                                      "assert_ade/eval_assert_ade_ade_corpus.csv"],
                       "ner_ade": ["ner_ade/eval_ner_ade_smm4h_task2.csv",
                                   "ner_ade/eval_ner_ade_cadec.csv",
                                   "ner_ade/eval_ner_ade_ade_corpus.csv"],
                       "ner_drug": ["ner_drug/eval_ner_drug_smm4h_task2.csv",
                                   "ner_drug/eval_ner_drug_cadec.csv",
                                   "ner_drug/eval_ner_drug_ade_corpus.csv"],
                       "ner_dosgae": ["ner_dosage/eval_ner_dosage_ade_corpus.csv"]
                       
                       }
    '''
    
    '''
    Model hyper-parameters, for detailed list of hyperparamets checkout model_args.py and global_args.py in /models/config
    The paramters description can also be found on https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    '''
    model_args = {
        "balancing_approach": "TDB",  # choose from TDB or TB
        "mixing_strategy": "TS", # choose from Proportional Mixing (PM) or Temperature Scaling (TS)
        "temperature_value": 2, #temperature value if using TS
        "dataset_path": os.getcwd() + '/data/combiner_data/',  #path to where the dataset is located
        "max_seq_length": 500,
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "num_train_epochs": 5,
        "evaluate_during_training": False,
        #"evaluate_during_training_steps": 5000, 
        #"evaluate_during_training_verbose": True,
        "max_length": 150,
        "learning_rate": 1e-4,
        "n_gpu": 4,
        "evaluate_generated_text": True,
        "gradient_accumulation_steps": 1,
        
        "use_multiprocessing": False,
        "fp16": True,
        
        
        "save_steps": -1,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,    "reprocess_input_data": True,
        "overwrite_output_dir": True,
        
        #"weight_decay": 0.01,
        #"warmup_steps": 600,
        
        "wandb_project": None
    }
    
    
    #Training the T5 model in a multi-task setting using the arguments defined above
    T5_Multi_Task_Train(train_task_dict, "t5-base", model_args)
    
    
    
if __name__ == "__main__":
    main()