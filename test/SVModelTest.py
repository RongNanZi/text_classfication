import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
import pickle as p
import os
import logging


logging.basicConfig(filename = 'svm_test.log', level = logging.INFO)

data_path = "../data/"
train = pd.read_csv(data_path+'train_set.csv', nrows=2000)
test = pd.read_csv(data_path+'test_set.csv',nrows=1)

logging.info("================================\n the csv file is read \n")

vect = TfidfVectorizer(ngram_range=(1,3), max_df=0.9, min_df= 3, smooth_idf=1, sublinear_tf=1)
train_num = vect.fit_transform(train['article'])
train_label = np.asarray(train['class'], dtype=int)
clf = SVC()

param_grid = [
    {"kernel":["linear","rbf"]},
    {'C':np.logspace(-2, 1, 2)},
    {'gamma':np.logspace(-3, 2, 2)}
]
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(clf,param_grid=param_grid, cv=cv, n_jobs=-1)
grid.fit(train_num, train_label)

logging.info('================================\n the grid search is done! \n')
#save the grid search result
with open('./grid.result', 'wb') as f:
    p.dump(grid, f)

#predict the test csv
test_num = vect.transform(test)
e_svm = SVC(grid.best_params_)
wanted = e_svm.predict(test_num)
df_wanted = pd.DataFrame(wanted, columns=['class'])
df_wanted['id'] = range(len(wanted))
if os.path.exists('./result'):
    os.mkdir('./result')
df_wanted[['id','class']].to_csv('./result/best_svm.csv', index=False)
