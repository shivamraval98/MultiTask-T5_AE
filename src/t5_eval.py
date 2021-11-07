import pandas as pd
from sklearn.metrics import classification_report
from models.T5Model.t5_model import T5Model 
from evaluation.eval_ner_t5 import eval_ner, calc_score
import torch
import os

torch.multiprocessing.set_sharing_strategy('file_system')

'''
Function to read the testing data, the testing dataset 
test_data: Pandas DataFrame containing the 3 columns - `prefix`, `input_text`, `target_text`.
        - `prefix`: A string indicating the task to perform.
        - `input_text`: The input text sequence. `prefix` is automatically prepended to form the full input. (<prefix>: <input_text>)
        - `target_text`: The target sequence
        
Parameters
-----------
task_type (str): The task type for testing the model, choose from: assert_ade, ner_ade, ner_drug, ner_dosage
task_name (str): The task name to test the model, choose from: smm4h_task1, smm4h_task2, cadec, ade_corpus, web_radr, smm4h_french

Returns
---------
test_df (pandas DataFrame): contains the testing dataset
'''
def read_df(task_type, task_name):        
    test_df = pd.read_csv(os.getcwd() + "/data/combiner_data/" + task_type + "/test_" + task_type + "_" + task_name + ".csv")
    
    return test_df

'''
Custom compute metric function to evaluate NER tasks
'''
def compute_metric_ner(labels, predictions):
    pred_df = pd.DataFrame(columns=["gold_labels", "pred_labels"])
    pred_df['gold_labels'] = labels
    pred_df['pred_labels'] = predictions
    pred_df.to_csv("ner_preds.csv", index = False)
    
    return 1

'''
Function to load the trained T5 model and evaluate on the NER task
'''
def ner_eval(model_path, df):
    model = T5Model("t5", model_path, args = {"n_gpu": 1}, use_cuda = False)
    model.eval_model(df, metrics = compute_metric_ner)
    match_df = eval_ner("ner_preds.csv")
    calc_score(match_df)    
    
'''
Custom compute metric function to evaluate binary (assertion) tasks
'''
def compute_metric_assert(labels, predictions):
    report = classification_report(labels, predictions, output_dict = True)
    print(report)
    return report

'''
Function to load the trained T5 model and evaluate on binary (assertion) tasks
'''
def assert_eval(model_path, df):
    model = T5Model("t5", model_path, args = {"n_gpu": 1})
    model.eval_model(df, metrics = compute_metric_assert)
    
   
'''
def main():
    #The task type for testing the model, choose from: assert_ade, ner_ade, ner_drug, ner_dosage
    test_task_type = "ner_ade"
    #The task name to test the model, choose from: smm4h_task1, smm4h_task2, cadec, ade_corpus, web_radr, smm4h_french
    test_task_name = "smm4h_task2"
    #Pre-trained model path
    model_path = "./outputs/"
    #prepare the dataset
    test_df = read_df(test_task_type, test_task_name)
    
    if "assert" in test_task_type:
        assert_eval(model_path, test_df)
    else:
        ner_eval(model_path, test_df)
'''

def main():
    model_path = "C:/ae-detect/models/old_combiner_model/"
    test_df = pd.DataFrame()
    test_df['prefix'] = ["ner adr"]
    test_df['input_text'] = ["I had wild dreams and felt nauseated after taking Norco in the morning"]
    test_df['target_text'] = ["wild dreams"]
    
    ner_eval(model_path, test_df)
    
 
if __name__ == "__main__":
    main()