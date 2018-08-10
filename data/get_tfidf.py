import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as p
data_path = "./"

train_data = pd.read_csv(data_path+'train_set.csv',
                         usecols=['article','word_seg'])
test_data = pd.read_csv(data_path+'test_set.csv',
                        usecols=['article', 'word_seg'])
data = pd.concat([train_data, test_data], axis=0)

word_vect = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df= 5,
                            smooth_idf=1, sublinear_tf=1)
word_vect.fit(data['article'])
word_train_num = word_vect.transform(train_data['article'])
word_test_num = word_vect.transform(test_data['article'])

token_vect = TfidfVectorizer(max_df=0.9, min_df= 5, smooth_idf=1,
                             sublinear_tf=1)
token_vect.fit(data['word_seg'])
token_train_num = token_vect.transform(train_data['word_seg'])
token_test_num = token_vect.transform(test_data['word_seg'])

def save(data, f_name):
    with open(data_path+f_name, 'wb') as f:
        p.dump(data, f)
save(word_train_num, 'word_train.tfidf')
save(word_test_num, 'word_test.tfidf')
save(token_train_num, 'token_train.tfidf')
save(token_test_num, 'token_test.tfidf')