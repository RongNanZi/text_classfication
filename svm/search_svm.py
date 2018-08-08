import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
import pickle as p
import logging
data_path = "./data/"
train = pd.read_csv(data_path+'train_set.csv') 
logging.basicConfig(filename='grid_search.log', filemode='w',level=logging.INFO)
vect = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df= 3, smooth_idf=1, sublinear_tf=1)
token_vect = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df= 3, smooth_idf=1, sublinear_tf=1)
train_word_num = vect.fit_transform(train['article'])
train_seg_num = token_vect.fit_transform(train['word_seg'])
logging.info('the feature engine is over')
train_label = np.asarray(train['class'], dtype=int)
clf = LinearSVC(dual=False)

param_grid = [
        {'penalty':['l1','l2']},
        {'C':[0.01,0.1,1]},
        {'class_weight':['balanced']}
]
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
word_grid = GridSearchCV(clf,param_grid=param_grid, cv=cv, n_jobs=8)
word_grid.fit(train_word_num, train_label)

#save the grid search result
with open('./word_grid.result', 'wb') as f:
    p.dump(word_grid, f)
logging.info('the word-level linear svc grid search is over')
seg_grid = GridSearchCV(clf,param_grid=param_grid, cv=cv, n_jobs=8)
seg_grid.fit(train_seg_num, train_label)

#save the grid search result
with open('./seg_grid.result', 'wb') as f:
    p.dump(seg_grid,f)


'''
predict the test csv
test_num = vect.transform(test['article'])
e_svm = SVC(grid.best_params_)
wanted = e_svm.predict(test_num)
df_wanted = pd.DataFrame(wanted, columns=['class'])
df_wanted['id'] = range(len(wanted))
df_wanted[['id','class']].to_csv('./best_svm.csv', index=False)
'''
