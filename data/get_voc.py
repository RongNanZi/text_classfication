import pandas as pd
import numpy as np
import pickle

train_csv = pd.read_csv('./train_set.csv')
test_csv = pd.read_csv('./test_set.csv')

from collections import Counter

def get_all_token(l_df, columns):
    all_token = []
    for df in l_df:
        for item in df[columns]:
            for token in item.split():
               all_token.append(token)
    all_token = Counter(all_token)
    return all_token

all_token = get_all_token([train_csv,test_csv], "word_seg")
all_word = get_all_token([train_csv,test_csv], "article")

with open('./token.all', 'wb')as f:
    pickle.dump(all_token, f)
with open('./word.all', 'wb')as f:
    pickle.dump(all_word, f)

def get_voc(size, all_token, f_name):
    wanted_voc = {'unk':0}    
    wanted_voc.update(zip(dict(all_token.most_common(size)).keys(), range(1,size+1)))
    with open('./'+f_name, 'wb')as f: 
        pickle.dump(wanted_voc, f)
        print('the voc file is saved')

get_voc(40000, all_token, 'token.voc')
get_voc(len(all_word),all_word, 'word.voc')
