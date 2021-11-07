import pandas as pd
import os
import zipfile
from data.importer.base_loader import BaseLoader

'''
CADEC Class object to load and prepare all the CADEC related datasets
'''
class CADECLoader(BaseLoader):
    def __init__(self):
        super().__init__()
        #Extract the files and folders from the raw zip file
        with zipfile.ZipFile(self.dataset_path + "CADEC/CADEC.zip","r") as zip_ref:
            zip_ref.extractall(self.dataset_path + 'CADEC/')
            
        with zipfile.ZipFile(self.dataset_path + "CADEC/CADEC.v2.zip","r") as zip_ref:
            zip_ref.extractall(self.dataset_path + 'CADEC/')
        
        self.txt_path = self.dataset_path + 'CADEC/cadec/text/'
        self.txt_files_list = os.listdir(self.txt_path)
        
        self.annon_path = self.dataset_path + 'CADEC/cadec/original/'
        self.annon_files_list = os.listdir(self.annon_path)
        self.save_path = os.getcwd() + "/data/combiner_data/"
        self.prep_raw_data()
        self.split_list = ["train", "eval", "test"]
        
    '''
    Function to retrive the id from the filename
    Parameters
    -----------
    str_name (str): input filename
    
    Returns
    ----------
    str_name (str): the id extracted from the filename
    '''
    def get_id(self, str_name):
        return str_name[:len(str_name)-4]

    '''
    Function to read the input file
    Parameters
    -----------
    filename (str): input filename
    
    Returns
    -----------
    final_str (str): the whole input text retrived from the input file
    '''
    def read_input(self, filename):
        lines = []
        with open(self.txt_path + filename, 'r') as f:
            for line in f:
                lines.append(line.strip())
            
        
        final_str = ' '.join(lines).strip()
        
        return final_str
    
    '''
    Function to extract the adverse drug reactions annotations from the annotation file 
    Parameters
    ------------
    filename (str): input filename
    
    Returns
    -----------
    adr_list (string): returns all the adverse drug extractions extracted from the file seperated by a semicolon(; ) 
    '''
    def read_annon(self, filename):
        adr_list = []
        with open(self.annon_path + filename, 'r') as f:
            for line in f:
                split_list = line.split('\t')
                if 'ADR' in split_list[1]:
                    adr_list.append(split_list[-1].strip())
        
        if len(adr_list) == 0:
            return 'none'
        else:
            return '; '.join(adr_list)
                
    '''
    Function to extract the drug mentions from the annotation file 
    Parameters
    ------------
    filename (str): input filename
    
    Returns
    -----------
    adr_list (string): returns all the drug mentions extracted from the file seperated by a semicolon(; ) 
    '''
    def read_annon_drug(self, filename):
        adr_list = []
        with open(self.annon_path + filename, 'r') as f:
            for line in f:
                split_list = line.split('\t')
                if 'Drug' in split_list[1]:
                    adr_list.append(split_list[-1].strip().lower())
        
        if len(adr_list) == 0:
            return 'none'
        elif len(adr_list) == 1:
            return adr_list[0]
        else:
            return '; '.join(set(adr_list))
    '''
    Herlper Function to prepare the raw data from the annotations file before splitting
    '''    
    def raw_data_helper(self, func_name, prefix_word, save_name):
        final_df = pd.DataFrame(columns=['Id', 'prefix', 'input_text', 'target_text'])
        id_list = []
        input_list = []
        target_list = []
        for i in range(len(self.txt_files_list)):
            samp_id = self.get_id(self.txt_files_list[i])
            id_list.append(samp_id)
            input_list.append(self.read_input(self.txt_files_list[i]))
            target_list.append(func_name(samp_id + '.ann'))
            
        final_df['Id'] = id_list
        final_df['prefix'] = [prefix_word]*len(id_list)
        final_df['input_text'] = input_list
        final_df['target_text'] = target_list
        final_df.to_csv(self.save_path + save_name, index = None)    

    '''
    Function to prepare the raw CADEC data
    '''
    def prep_raw_data(self):
        self.raw_data_helper(self.read_annon, "ner ade", "ner_ade/raw_data_cadec.csv")         
        self.raw_data_helper(self.read_annon_drug, "ner drug", "ner_drug/raw_data_cadec.csv") 

    def read_ids(self, filename):
        with open(self.split_path + 'CADEC/' + filename, 'r') as f:
            str1 = f.read()
        
        return str1.split('\n')
    
    '''
    Function to prepare the AE Detection Dataset for CADEC
    '''    
    def prep_assert_ade(self):
        raw_data = pd.read_csv(self.save_path + "ner_ade/raw_data_cadec.csv", index_col = [0])
        for split_name in self.split_list:
            ids = self.read_ids(split_name + '.txt')
            input_list = []
            target_list = []
            for i in range(len(ids)):
                input_list.append(raw_data.loc[ids[i]]['input_text'])
                target_list.append(raw_data.loc[ids[i]]['target_text'])
                
            tgt_txt = []
            for samp in target_list:
                if samp == "none":
                    tgt_txt.append("healthy okay")
                else:
                    tgt_txt.append("adverse event problem")
                    
            new_df = pd.DataFrame(columns = ['prefix', 'input_text', 'target_text'])
            new_df['prefix'] = ['assert ade']*len(input_list)
            new_df['input_text'] = input_list
            new_df['target_text'] = tgt_txt
            
            new_df.to_csv(self.save_path + "assert_ade/" + split_name + "_assert_ade_cadec.csv", index = None)
            
        print("CADEC AE Detection dataset Saved Successfully!")
            
    '''
    Function to prepare the AE Extraction Dataset for CADEC
    '''
    def prep_ner_ade(self):
        raw_data = pd.read_csv(self.save_path + "ner_ade/raw_data_cadec.csv", index_col = [0])
        for split_name in self.split_list:
            ids = self.read_ids(split_name + '.txt')
            input_list = []
            target_list = []
            for i in range(len(ids)):
                input_list.append(raw_data.loc[ids[i]]['input_text'])
                target_list.append(raw_data.loc[ids[i]]['target_text'])
                
            new_df = pd.DataFrame(columns = ['prefix', 'input_text', 'target_text'])
            new_df['prefix'] = ['ner ade']*len(input_list)
            new_df['input_text'] = input_list
            new_df['target_text'] = target_list
            
            new_df.to_csv(self.save_path + "ner_ade/" + split_name + "_ner_ade_cadec.csv", index = None)
        
        print("CADEC AE Extraction dataset Saved Successfully!")
            
    '''
    Function to prepare the Drug Extraction Dataset for CADEC
    '''
    def prep_ner_drug(self):
        raw_data = pd.read_csv(self.save_path + "ner_drug/raw_data_cadec.csv", index_col = [0])
        for split_name in self.split_list:
            ids = self.read_ids(split_name + '.txt')
            input_list = []
            target_list = []
            index_keys = raw_data.index.tolist()
            for i in range(len(ids)):
                if ids[i] in index_keys:
                    input_list.append(raw_data.loc[ids[i]]['input_text'])
                    target_list.append(raw_data.loc[ids[i]]['target_text'])
                
            new_df = pd.DataFrame(columns = ['prefix', 'input_text', 'target_text'])
            new_df['prefix'] = ['ner drug']*len(input_list)
            new_df['input_text'] = input_list
            new_df['target_text'] = target_list
            
            new_df.to_csv(self.save_path + "ner_drug/" + split_name + "_ner_drug_cadec.csv", index = None)

        print("CADEC Drug Extraction dataset Saved Successfully!")
            