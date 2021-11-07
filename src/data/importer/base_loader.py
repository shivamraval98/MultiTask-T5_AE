import pandas as pd
import os
from sklearn.model_selection import train_test_split
import preprocessor as p

'''
BaseLoader class which is extended by other Data Loaders
'''
class BaseLoader(object):
    def __init__(self):
        self.split_path = os.getcwd() + "/data/splits/"
        self.dataset_path = os.getcwd() + "/data/datasets/"
        
    '''
    Function to lowercase all the strings in the list
    Parameters
    --------------
    samp_list (list): input list on which the lowercases operation needs to be performed
    
    Returns
    --------------
    lowercase_str (list): return list of the strings after performing lowercase operation
    '''    
    def str_lower_helper(self, samp_list):
        lowercase_str = []
        for samp in samp_list:
            if type(samp) == type("str"):
                lowercase_str.append(samp.lower())
            else:
                lowercase_str.append(samp)
                
        return lowercase_str
    
    '''
    Function to preprocess the given tweet sentence to remove URL, EMOJI and SMILEY
    Parameters
    ---------------
    raw_str (str): the input string which needs to be pre-processed
    
    Returns
    -------------
    clean_str (str): returns the processed string
    '''
    def twitter_preprocess(self, raw_str):
        p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY)
        clean_str = ""

        try:
            clean_str = p.clean(raw_str)
        except:
            clean_str = raw_str
                    
        return clean_str
