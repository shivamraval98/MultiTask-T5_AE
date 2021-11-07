import torch
from models.bert_dataset_loader import *
from models.bert_model import *
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

'''
Function to test the trained model on a given dataset
Parameters
------------
model_name (str): pre-trained model name from the huggingface transformers library
model_path (str): the path to the trained model
task_name (str): task name (ex. smm4h_task1, cadec ..)
'''
def test_model(model_name, model_path, task):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_obj = BaselineBERT(model_name, task)
    model = BertForSequenceClassification.from_pretrained(model_path)
    _,_, test_dataset = model_obj.get_torch_dataset()
    
    model.to(device)
    model.eval()
    preds_list = []
    labels_list = []
    for i in range(len(test_dataset)):
        b1 = test_dataset[i]
        outputs = model(b1['input_ids'].unsqueeze(0).to(device))
        pred = outputs.logits.argmax(-1)
        preds_list.append(pred.item())
        labels_list.append(b1['labels'].item())

    print("Test Results")
    report = classification_report(labels_list, preds_list, output_dict = True)    
    print(report)
    
    
def main():
    #choose any pre-trained BERT based models from huggingface's transformers library (ex: dmis-lab/biobert-v1.1, allenai/scibert_scivocab_uncased, ...)
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    #Path to the saved model
    model_path = "./baseline"
    #choose from the following assert_ade task to train the model (smm4h_task1, smm4h_task2, cadec, ade_corpus)
    task = "smm4h_task1"
    
    #Testing the model on the given pre-trained model and task
    test_model(model_name, model_path, task)
    
if __name__ == "__main__":
    main()
    
