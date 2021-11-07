import pandas as pd
import os
from data.importer.base_loader import BaseLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split

class ADECorpusLoader(BaseLoader):
    def __init__(self):
        super().__init__()
        self.random_seed = 42
        self.save_path = os.getcwd() + "/data/combiner_data/"
        
    '''
    Function to partition the given data into train, eval and test
    Parameters
    -----------
    data (pandas dataframe): input data which needs to be partitioned
    
    Returns
    ----------
    train_df (pandas dataframe): train split from the input data (60%)
    eval_df (pandas dataframe): eval split from the input data (20%)
    test_df (pandas dataframe): test split from the input data (20%)
    '''   
    def get_splits(self, data):
        train_df, eval_df = train_test_split(data, test_size = 0.4, random_state = self.random_seed)
        eval_df, test_df = train_test_split(eval_df, test_size = 0.5, random_state = self.random_seed)
        
        return train_df, eval_df, test_df
    
    '''
    Function to prepare the AE Detection Dataset for ADE Corpus v2
    '''
    def prep_assert_ade(self):
        dataset = load_dataset('ade_corpus_v2', 'Ade_corpus_v2_classification', split = 'train')
        
        tgt_txt = []
        for samp in dataset['label']:
            if samp == 0:
                tgt_txt.append("healthy okay")
            else:
                tgt_txt.append("adverse event problem")
        
        df = pd.DataFrame()
        df['prefix'] = ["assert ade"]*len(dataset['label'])
        df['input_text'] = dataset['text']
        df['target_text'] = tgt_txt
        
        pos_data = df[df['target_text'] == "adverse event problem"]
        neg_data = df[df['target_text'] == "healthy okay"]
        
        pos_train, pos_eval, pos_test = self.get_splits(pos_data)
        neg_train, neg_eval, neg_test = self.get_splits(neg_data)
        
        split_dict = {"train": [pos_train, neg_train], "eval": [pos_eval, neg_eval], "test": [pos_test, neg_test]}
        for key in list(split_dict.keys()):
            res_df = pd.concat(split_dict[key])
            res_df = res_df.sample(frac = 1)
            
            res_df.to_csv(self.save_path + "assert_ade/" + key + "_assert_ade_ade_corpus.csv", index = None)
        
        print("ADE Corpus AE Detection dataset Saved Successfully!")
            
    '''
    Function to prepare the AE Extraction Dataset for ADE Corpus v2
    '''
    def prep_ner_ade(self):
        dataset = load_dataset('ade_corpus_v2', 'Ade_corpus_v2_drug_ade_relation', split = 'train')

        df = pd.DataFrame()
        df['prefix'] = ["ner ade"]*len(dataset['text'])
        df['input_text'] = dataset['text']
        df['target_text']  = dataset['effect']

        train_df, eval_df, test_df = self.get_splits(df)
        split_dict = {"train": train_df, "eval": eval_df, "test": test_df}
        for key in list(split_dict.keys()):
            res_df = split_dict[key]            
            res_df.to_csv(self.save_path + "ner_ade/" + key + "_ner_ade_ade_corpus.csv", index = None)
            
        print("ADE Corpus AE Extraction dataset Saved Successfully!")
            
    '''
    Function to prepare the Drug Extraction Dataset for ADE Corpus v2
    '''    
    def prep_ner_drug(self):
        dataset_drug_ade = load_dataset('ade_corpus_v2', 'Ade_corpus_v2_drug_ade_relation', split = 'train')
        dataset_drug_dosage = load_dataset('ade_corpus_v2', 'Ade_corpus_v2_drug_dosage_relation', split = 'train')
        
        df = pd.DataFrame()
        df['prefix'] = ["ner drug"]*(len(dataset_drug_ade['text']) + len(dataset_drug_dosage['text']))
        df['input_text'] = dataset_drug_ade['text'] + dataset_drug_dosage['text']
        df['target_text']  = dataset_drug_ade['drug'] + dataset_drug_dosage['drug']

        train_df, eval_df, test_df = self.get_splits(df)
        split_dict = {"train": train_df, "eval": eval_df, "test": test_df}
        for key in list(split_dict.keys()):
            res_df = split_dict[key]            
            res_df.to_csv(self.save_path + "ner_drug/" + key + "_ner_drug_ade_corpus.csv", index = None)
            
        print("ADE Corpus Drug Extraction dataset Saved Successfully!")
            
    '''
    Function to prepare the Drug Dosage Extraction Dataset for ADE Corpus v2
    '''
    def prep_ner_dosage(self):
        dataset = load_dataset('ade_corpus_v2', 'Ade_corpus_v2_drug_dosage_relation', split = 'train')

        df = pd.DataFrame(columns = ['prefix', 'input_text', 'target_text'])
        
        df['prefix'] = ['ner dosage']*len(dataset['text'])
        df['input_text'] = dataset['text']
        df['target_text'] = dataset['dosage']
        
        train_df, eval_df, test_df = self.get_splits(df)
        split_dict = {"train": train_df, "eval": eval_df, "test": test_df}
        for key in list(split_dict.keys()):
            res_df = split_dict[key]            
            res_df.to_csv(self.save_path + "ner_dosage/" + key + "_ner_dosage_ade_corpus.csv", index = None)
            
        print("ADE Corpus Drug Dosage Extraction dataset Saved Successfully!")
        
                
        
    