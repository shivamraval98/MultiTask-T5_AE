import pandas as pd
import os
import torch

'''
Dataset object to load the dataset for the BERT based models
'''
class Dataset_obj(object):
    def __init__(self, task):
        self.data_path = os.getcwd() + "/data/combiner_data/assert_ade/"
        self.task = task
        self.data_dict = {}

    '''
    Function to prepare the input and the labels
    Parameters
    ------------
    df (pandas dataframe): input data
    
    Returns
    -----------
    input_text (list): all the input from the dataframe
    labels (lists): binary labels for input to the BERT model
    '''
    def prep_assert_data(self, df):
        input_text = df['input_text'].tolist()
        target_text = df['target_text'].tolist()
        
        labels = []
        for samp in target_text:
            if samp == "healthy okay":
                labels.append(0)
            else:
                labels.append(1)
        
        return input_text, labels

    '''
    Function to prepare the data for train, eval and test split
    '''
    def get_data_dict(self): 
        train_text, train_labels = self.prep_assert_data(pd.read_csv(self.data_path + "train_assert_ade_" + self.task + ".csv").astype(str))
        pos_weight = sum(train_labels)/len(train_labels)
        
        eval_text, eval_labels = self.prep_assert_data(pd.read_csv(self.data_path + "eval_assert_ade_" + self.task + ".csv").astype(str))
        test_text, test_labels = self.prep_assert_data(pd.read_csv(self.data_path + "test_assert_ade_" + self.task + ".csv").astype(str))
        

        class_weights = torch.FloatTensor([pos_weight, 1 - pos_weight])
        
        self.data_dict['train_text'] = train_text
        self.data_dict['train_labels'] = train_labels
        self.data_dict['eval_text'] = eval_text
        self.data_dict['eval_labels'] = eval_labels 
        self.data_dict['test_text'] = test_text
        self.data_dict['test_labels'] = test_labels
        
        self.data_dict['class_weights'] = class_weights
        
        return self.data_dict
    
