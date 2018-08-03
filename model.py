from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.svm import SVC
import numpy as np
from sklearn import metrics

data_path = "D:/home/acgotaku/Downloads/new_data/"
train, val = train_test_split(pd.read_csv(data_path+'train_set.csv')[['word_seg','class']], test_size= 0.2)

vect = TfidfVectorizer(ngram_range=(1,2), max_df=0.8, min_df= 5)

train_num = vect.fit_transform(train['word_seg'])
val_num = vect.transform(val['word_seg'])
train_label = np.asarray(train['class'], type=int)
val_label = np.asarray(val['class'], type=int)

clfs = [svm.LinearSVC().fit(train_num, train_label), SVC(kernel='rbf', gamma=5, C=0.001).fit(train_num, train_label)]

for i in clfs:
    pred = clfs[i].predict(val_num)
    print("this is {0} calssification, \
    the pecision is {1},\
          the recall is {2} \
           the f1_score is {3}").format(metrics.precision_score(val_label, pred),\
                                        metrics.recall_score(val_label, pred),\
                                        metrics.f1_score(val_label, pred))