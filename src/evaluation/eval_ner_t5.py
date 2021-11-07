import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz


'''
Function to check the string matching
Parameters
---------------
gold_word: gold label string
pred_word: predicted label string
'''
def single_check(gold_word, pred_word):
    #Splitting the words
    gold_split = gold_word.split(" ")
    pred_split = pred_word.split(" ")
    
    #Case 1: Same Length after splitting
    if len(gold_split) == len(pred_split):
        #Subcase 1.1: Strings are same, return partial:COR, exact:COR, strict:COR
        if gold_word == pred_word:
            return "COR", "COR", "COR"
        else:
            t_cnt = 0
            for idx in range(len(gold_split)):
                if gold_split[idx] == pred_split[idx]:
                    t_cnt = t_cnt + 1
            #Subcase 1.2: None of the word is same, return partial:INC, exact:INC, strict:INC
            if t_cnt == 0:
                return "INC", "INC", "INC"
            #Subcase 1.3: Atleast one word matches with the gold label, return partial:PAR, exact:INC, strict:INC
            else:
                return "PAR", "INC", "INC"
    #Case 2: Length of gold label is greater than predicted label (i.e prediction is partial or incorrect)
    elif len(gold_split) > len(pred_split):
        t_cnt = 0
        for idx in range(len(pred_split)):
            if pred_split[idx] in gold_word:
                t_cnt = t_cnt + 1
        #Subcase 2.1: None of the word is same, return partial:INC, exact:INC, strict:INC
        if t_cnt == 0:
            return "INC", "INC", "INC"
        #Subcase 2.2: Atleast one word matches with the gold label, return partial:PAR, exact:INC, strict:INC
        else:
            return "PAR", "INC", "INC"
    #Case 3: Length of gold label is smaller than predicted label (i.e prediction is partial or incorrect)
    elif len(gold_split) < len(pred_split):
        t_cnt = 0
        for idx in range(len(gold_split)):
            if gold_split[idx] in pred_word:
                t_cnt = t_cnt + 1
        #Subcase 3.1: None of the word is same, return partial:INC, exact:INC, strict:INC
        if t_cnt == 0:
            return "INC", "INC", "INC"
        #Subcase 3.2: Atleast one word matches with the gold label, return partial:PAR, exact:INC, strict:INC
        else:
            return "PAR", "INC", "INC"

'''
NER spans extracted by T5 can be in a different order so implementing a novel fuzzy based approach to extract string based on scores and calculating scores
Parameters
--------------
gold_split: multiple gold label spans seperated by ;
pred_split: multiple predicted label spans seperated by ;


Returns
-------------
g_words: indivdual gold label spans
p_words: indivdual predicted label spans
partial: score labels for partial matching
excat: score labels for exact matching
strict: score labels for strict matching
'''
def fuzzy_approach(gold_split, pred_split):
    g_words = []
    p_words = []
    partial = []
    exact = []
    strict = []
    g_del = []
    p_del = []
    #Calculate fuzzy matrix
    fuzz_mat = np.zeros([len(gold_split), len(pred_split)])
    for i in range(fuzz_mat.shape[0]):
        for j in range(fuzz_mat.shape[1]):
            fuzz_mat[i][j] = fuzz.ratio(gold_split[i], pred_split[j])
    
    min_len = min(len(gold_split), len(pred_split))
    
    #Iterating over the min len of the matrix 
    for idx in range(min_len):
        max_ind = np.unravel_index(np.argmax(fuzz_mat, axis=None), fuzz_mat.shape)
        row, col = max_ind[0], max_ind[1]
        g_del.append(row)
        p_del.append(col)
        #Case 1: Exact Match
        if gold_split[row] == pred_split[col]:
            g_words.append(gold_split[row])
            p_words.append(pred_split[col])
            partial.append("COR")
            exact.append("COR")
            strict.append("COR")
            fuzz_mat[row,:] = -1
            fuzz_mat[:,col] = -1
        else:
            g_samp = gold_split[row]
            p_samp = pred_split[col]
            #Case 2: handle the subcases in single_check function
            s1,s2,s3 = single_check(g_samp, p_samp)
            g_words.append(g_samp)
            p_words.append(p_samp)
            partial.append(s1)
            exact.append(s2)
            strict.append(s3)
            fuzz_mat[row,:] = -1
            fuzz_mat[:,col] = -1
    
    #After extracting strings either there are more spans in predictions or gold label
    if len(gold_split) > len(pred_split):
        #Delete words for which the scores are already calculated
        remove_words = [gold_split[del_idx] for del_idx in g_del]
        for words in remove_words:
            gold_split.remove(words)
        
        #For the remaining gold spans there is no labels predicted so missing
        for rem_words in gold_split:
            g_words.append(rem_words)
            p_words.append("none")
            partial.append("MIS")
            exact.append("MIS")
            strict.append("MIS")
        
        return g_words, p_words, partial, exact, strict
            
    elif len(gold_split) < len(pred_split):
        #Delete words for which the scores are already calculated
        remove_words = [pred_split[del_idx] for del_idx in p_del]
        for words in remove_words:
            pred_split.remove(words)
        
        #For the remaining predicted spans there is no corresponding gold label so they are extra (spurious)
        for rem_words in pred_split:
            g_words.append("none")
            p_words.append(rem_words)
            partial.append("SPU")
            exact.append("SPU")
            strict.append("SPU")
        
        return g_words, p_words, partial, exact, strict
            
    
    return g_words, p_words, partial, exact, strict


'''
Function to give scores for each predictions and preparing a dataframe
Parameters
-------------
filename: file generated as a predictions from the T5 model

Returns
------------
A dataframe containing tag for partial, exact and strict matching to calculate scores
'''                                    
def eval_ner(filename):
    data_res = pd.read_csv(filename)
    gold_list = data_res['gold_labels'].tolist()
    pred_list = data_res['pred_labels'].tolist()
    
    assert len(gold_list) == len(pred_list)

    partial = []
    exact = []
    strict = []
    gold_df_list = []
    pred_df_list = []
    for i in range(len(gold_list)):
        gold = gold_list[i].lower()
        pred = pred_list[i].lower()
        

        if gold == "none":
            gold_split = []
        else:
            gold_split = gold.split('; ')
            
        if pred == "none":
            pred_split = []
        else:
            pred_split = pred.split('; ')
        
        #Case 1: No spans predicted, score is missing for all matching        
        if len(pred_split) == 0 and len(gold_split) != 0:
            partial.append("MIS")
            exact.append("MIS")
            strict.append("MIS")
            gold_df_list.append(gold)
            pred_df_list.append(pred)
        #Case 2: Gold spans are empty but predictions have spans, score is spurious for all matching
        elif len(gold_split) == 0 and len(pred_split) != 0:
            partial.append("SPU")
            exact.append("SPU")
            strict.append("SPU")
            gold_df_list.append(gold)
            pred_df_list.append(pred)
        #Case 3: negative class, no spans extracted in predictions and no span in gold labels
        elif len(gold_split) == 0 and len(pred_split) == 0:
            partial.append("COR")
            exact.append("COR")
            strict.append("COR")
            gold_df_list.append(gold)
            pred_df_list.append(pred)
        #Case 4: Handle all other cases using fuzzy approach when there are spans in gold and predict
        else:
            g_add, p_add, s1, s2, s3 = fuzzy_approach(gold_split, pred_split)
            gold_df_list += g_add
            pred_df_list += p_add
            partial += s1
            exact += s2
            strict += s3
            
            
    match_df = pd.DataFrame(columns=['Gold Label', 'Pred Label', 'Partial', 'Exact', 'Strict'])
    match_df['Gold Label'] = gold_df_list
    match_df['Pred Label'] = pred_df_list
    match_df['Partial'] = partial
    match_df['Exact'] = exact
    match_df['Strict'] = strict
    

    return match_df

#Calculate scores
def cnt_labels(label_list, label_type):
    score_dict = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurius': 0}
    for i in range(len(label_list)):
        if label_list[i] == "COR":
            score_dict['correct'] += 1
        elif label_list[i] == "INC":
            score_dict['incorrect'] +=  1
        elif label_list[i] == "PAR":
            score_dict['partial'] +=  1
        elif label_list[i] == "MIS":
            score_dict['missed'] +=  1
        elif label_list[i] == "SPU":
            score_dict['spurius'] += 1
            
    POS = score_dict['correct'] + score_dict['incorrect'] + score_dict['partial'] + score_dict['missed']
    ACT = score_dict['correct'] + score_dict['incorrect'] + score_dict['partial'] + score_dict['spurius']
    
    if POS == 0 or ACT == 0:
        raise ValueError("No Strict spans Extracted")
    
    if label_type == "partial":
        precision = (score_dict['correct'] + (0.5*score_dict['partial']))/ACT
        recall = (score_dict['correct'] + (0.5*score_dict['partial']))/POS
    else:
        precision = score_dict['correct']/ACT
        recall = score_dict['correct']/POS
        
    f1_score = 2*((precision*recall)/(precision+recall))
    
    score_dict['precision'] = precision
    score_dict['recall'] = recall
    score_dict['f1_score'] = f1_score
    
    return score_dict
    
#Calculate and print final scores for every matching
def calc_score(df):
    final_scores = {}
    final_scores['partial'] = cnt_labels(df['Partial'].tolist(), 'partial')
    final_scores['exact'] = cnt_labels(df['Exact'].tolist(), 'exact')
    final_scores['strict'] = cnt_labels(df['Strict'].tolist(), 'strict')
    
    
    print(final_scores)
        
        
