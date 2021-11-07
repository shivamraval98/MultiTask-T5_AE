import pandas as pd
import os
from data.importer.base_loader import BaseLoader
from sklearn.model_selection import train_test_split

'''
SMM4H Class object to load and prepare all the SMM4H related datasets
'''
class SMM4HLoader(BaseLoader):
    def __init__(self):
        super().__init__()
        self.extraction_data = pd.read_csv(self.dataset_path + 'SMM4H_Task2/SMM4H19_Task2.csv', index_col="tweet_id")
        self.save_path = os.getcwd() + "/data/combiner_data/"
        
    '''
    Function to get the split ids of SMM4H train, eval or test from the splits folder
    Parameters
    ------------
    filename (str): the input filename to fetch the splits can be train, eval or test
    
    Returns
    ------------
    l1 (list): return the ids from that split
    '''
    def get_split_ids(self, filename):
        with open(self.split_path + "SMM4H_Task2/" + filename, 'r') as f:
            l1 = f.read()
            
        return l1.split('\n')
    
    '''
    Function to get input and target list based on the ids
    Parameters
    ------------
    split_id (list): list containing the id from which samples are extracted
    extraction_term (str): either extraction or drug keyword depedning upon the dataset wihch is prepared
    
    Returns
    ------------
    tweet (list): extracted input tweets according to ids
    extraction (list): extracted target strings according to ids
    '''
    def get_input_target(self, split_id, extraction_term):
        tweet = []
        extraction = []
        
        for ids in split_id:
            search_id = ids
            row_data = self.extraction_data.loc[search_id]
            
            try:
                for idx in range(self.extraction_data.loc[search_id].shape[0]):
                    tweet.append(self.twitter_preprocessor(row_data.iloc[idx]['tweet']))
                    ex_term = row_data.iloc[idx][extraction_term]
                    if type(ex_term) != type("hi"):
                        extraction.append('none')
                    else:
                        extraction.append(ex_term.lower())
            except:
                tweet.append(row_data['tweet'])
                ex_term = row_data[extraction_term]
                
                if type(ex_term) != type("hi"):
                    extraction.append('none')
                else:
                    extraction.append(ex_term.lower())
                    
                    
        return tweet, extraction
        
    '''
    Function to prepare the AE Extraction dataset for SMM4H Task 2
    '''
    def prep_data_ner_ade_smm4h_task2(self):
        split_list = ["train", "eval", "test"]
        for split in split_list:
            ids = self.get_split_ids(split + '.txt')
            tweet, extraction = self.get_input_target(ids, "extraction")
            res_data = pd.DataFrame()
            res_data['prefix'] = ["ner ade"]*len(tweet)
            res_data['input_text'] = tweet
            res_data['target_text'] = extraction
            res_data = res_data.sample(frac=1)
        
            res_data.to_csv(self.save_path + 'ner_ade/' + split + '_ner_ade_smm4h_task2.csv', index = None)
        
        print("SMM4H Task2 AE Extraction dataset Saved Successfully!")
    
    '''
    Function to prepare the Drug Extraction Dataset for SMM4H Task 2
    '''
    def prep_data_ner_drug_smm4h_task2(self):
        split_list = ["train", "eval", "test"]
        for split in split_list:
            ids = self.get_split_ids(split + '.txt')
            tweet, extraction = self.get_input_target(ids, "drug")
            res_data = pd.DataFrame()
            res_data['prefix'] = ["ner drug"]*len(tweet)
            res_data['input_text'] = tweet
            res_data['target_text'] = extraction
            res_data = res_data.sample(frac=1)
        
            res_data.to_csv(self.save_path + 'ner_drug/' + split + '_ner_drug_smm4h_task2.csv', index = None)
            
        print("SMM4H Task2 Drug Extraction dataset Saved Successfully!")
            
    '''
    Function to partition the given data into train, eval and test
    Parameters
    -----------
    data (pandas dataframe): input data which needs to partitioned
    
    Returns
    ----------
    data_train (pandas dataframe): train split from the input data (80%)
    data_eval (pandas dataframe): eval split from the input data (10%)
    data_test (pandas dataframe): test split from the input data (10%)
    '''   
    def split_parts(self, data):
        data_train, data_eval = train_test_split(data, test_size = 0.2, shuffle = True)
        data_eval, data_test = train_test_split(data_eval, test_size = 0.5, shuffle = True)
        
        return data_train, data_eval, data_test
    
    '''
    Function to restructure the data and save 
    Parameters
    -------------
    df (pandas dataframe): the input dataframe which needs to be restructured
    save_name (str): the name of the csv file to be saved
    '''
    def restructure_data(self, df, save_name):
        res_df = pd.DataFrame()
        label = df["label"].tolist()
        tgt_txt = []
        for val in label:
            if val == 0:
                tgt_txt.append("healthy okay")
            else:
                tgt_txt.append("adverse event problem")
                
        res_df["prefix"] = ["assert ade"]*len(df)
        
        res_df["input_text"] = [self.twitter_preprocess(in_txt) for in_txt in df['tweet'].tolist()]
        res_df["target_text"] = tgt_txt
        
        res_df.to_csv(self.save_path + save_name, index = None)
        
    '''
    Function to prepare the AE Detection Dataset for SMM4H Task 1
    '''
    def prep_data_assert_ade_task1(self):
        raw_data = pd.read_csv(self.dataset_path + "SMM4H_Task1/SMM4H19_Task1.csv")
        pos_data = raw_data[raw_data["label"] == 1]
        neg_data = raw_data[raw_data["label"] == 0]
        
        pos_data_train, pos_data_eval, pos_data_test = self.split_parts(pos_data)
        neg_data_train, neg_data_eval, neg_data_test = self.split_parts(neg_data)
        
        train_df = pd.concat([pos_data_train, neg_data_train])
        train_df = train_df.sample(frac = 1)
        
        eval_df = pd.concat([pos_data_eval, neg_data_eval])
        eval_df = eval_df.sample(frac = 1)
        
        test_df = pd.concat([pos_data_test, neg_data_test])
        test_df = test_df.sample(frac = 1)
        
        self.restructure_data(train_df, 'assert_ade/train_assert_ade_smm4h_task1.csv')
        self.restructure_data(eval_df, 'assert_ade/eval_assert_ade_smm4h_task1.csv')
        self.restructure_data(test_df, 'assert_ade/test_assert_ade_smm4h_task1.csv')
        
        print("SMM4H Task1 AE Detection dataset Saved Successfully!")

    '''
    Function to prepare the AE Detection Dataset for SMM4H Task 2
    '''
    def prep_data_assert_ade_task2(self):
        split_list = ["train", "eval", "test"]
        for split in split_list:
            ids = self.get_split_ids(split + '.txt')
            tweet, extraction = self.get_input_target(ids, "extraction")
            label = []
            for samp in extraction:
                if samp == "none":
                    label.append("healthy okay")
                else:
                    label.append("adverse event problem")
            res_data = pd.DataFrame()
            res_data['prefix'] = ["ner ade"]*len(tweet)
            res_data['input_text'] = tweet
            res_data['target_text'] = label
            res_data = res_data.sample(frac=1)
        
            res_data.to_csv(self.save_path + 'assert_ade/' + split + '_assert_ade_smm4h_task2.csv', index = None)
            
        print("SMM4H Task2 AE Detection dataset Saved Successfully!")
            
    '''
    Function to prepare the AE Detection Dataset for SMM4H French
    '''
    def prep_data_assert_ade_french(self):
        french_df = pd.read_csv(self.dataset_path + 'SMM4H_French/SMM4H_French.csv')
        self.restructure_data(french_df, 'french_data/test_assert_ade_smm4h_french.csv')
        
        print("SMM4H French AE Detection dataset Saved Successfully!")
        
        
        