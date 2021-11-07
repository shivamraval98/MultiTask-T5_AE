import pandas as pd
import os
from data.importer.base_loader import BaseLoader


class WEBRADRLoader(BaseLoader):
    def __init__(self):
        super().__init__()
        self.raw_data = pd.read_csv(self.dataset_path + "WEB_RADR/WEB_RADR.csv")
        self.raw_input = self.raw_data["tweet"].tolist()
        self.in_txt = [self.twitter_preprocess(samp) for samp in self.raw_input]
        self.save_path = os.getcwd() + "/data/combiner_data/"
        
    '''
    Function to prepare the AE Detection Dataset for WEB-RADR
    '''
    def prep_assert_ade(self):
        raw_target = self.raw_data["label"].tolist()
        
        tgt_txt = []
        for samp in raw_target:
            if samp == 0:
                tgt_txt.append("healthy okay")
            else:
                tgt_txt.append("adverse event problem")
        
        res_df = pd.DataFrame()
        res_df["prefix"] = ["assert ade"]*len(self.in_txt)
        res_df["input_text"] = self.in_txt
        res_df["target_text"] = tgt_txt

        res_df.to_csv(self.save_path + "web_radr/test_assert_ade_web_radr_all.csv", index = None)    
        
        print("WEB-RADR AE Detection dataset Saved Successfully!")
        
    '''
    Function to prepare the AE Extraction Dataset for WEB-RADR
    '''
    def prep_ner_ade(self):
        raw_target = self.raw_data["extraction"].tolist()
        
        corr_labels = []
        for label in raw_target:
            if type(label) != type('str'):
                corr_labels.append('none')
            else:
                split_label = label.split(';')
                str_prep = '; '.join(split_label)
                corr_labels.append(str_prep)
                
        
        tgt_txt = [self.twitter_preprocess(samp) for samp in corr_labels]
        
        res_df = pd.DataFrame()
        res_df["prefix"] = ["ner ade"]*len(self.in_txt)
        res_df["input_text"] = self.in_txt
        res_df["target_text"] = tgt_txt
        
        res_df.to_csv(self.save_path + "web_radr/test_ner_ade_web_radr_all.csv", index = None)    
        
        
        print("WEB-RADR AE Extraction dataset Saved Successfully!")
    