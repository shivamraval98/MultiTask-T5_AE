#Importing required Libraries
import pandas as pd
import os
from data.importer.base_loader import *
from data.importer.SMM4H_loader import *
from data.importer.CADEC_loader import *
from data.importer.ADECorpus_loader import *
from data.importer.WEBRADR_loader import *
from data.importer.IRMS_loader import *

def main():
    #Various Dataset Loader objects 
    smm4h_obj = SMM4HLoader()
    cadec_obj = CADECLoader()
    adecorpus_obj = ADECorpusLoader()
    webradr_obj = WEBRADRLoader()
    
    #Loading all the AE Detection Datasets
    print("Loading Datasets...")
    print("Preparing AE Detection Datasets.....")
    smm4h_obj.prep_data_assert_ade_task1()
    smm4h_obj.prep_data_assert_ade_task2()
    cadec_obj.prep_assert_ade()
    adecorpus_obj.prep_assert_ade()
    webradr_obj.prep_assert_ade()
    smm4h_obj.prep_data_assert_ade_french()
    print("\n")
    
    #Laoding all the AE Extraction Datasets
    print("Preparing AE Extraction Datasets....")
    smm4h_obj.prep_data_ner_ade_smm4h_task2()
    cadec_obj.prep_ner_ade()
    adecorpus_obj.prep_ner_ade()
    webradr_obj.prep_ner_ade()
    print("\n")
    
    #Laoding all the Drug Extraction Datasets
    print("Preparing Drug Extraction Datasets....")
    smm4h_obj.prep_data_ner_drug_smm4h_task2()
    cadec_obj.prep_ner_drug()
    adecorpus_obj.prep_ner_drug()
    print("\n")
    
    #Loading the Drug Dosage dataset
    print("Preparing Drug Doage Extraction Dataset....")
    adecorpus_obj.prep_ner_dosage()
    print("\n")
    
        
if __name__ == "__main__":
    main()
    
