import pandas as pd
import pickle as p
import os
import numpy as np

data_path = '../data/'
log_path = './log/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
csv_data = pd.read_csv(data_path+'train_set.csv')
word_voc_file = '../data/word.voc'
token_voc_file = '../data/token.voc'
with open(word_voc_file, 'rb') as f:
    word_voc = p.load(f)
with open(token_voc_file, 'rb') as f:
    token_voc = p.load(f)
word_length = 10000
token_length = 7500

def token2num(text, voc, length):
    wanted = []
    text = text.split()
    for token in text:
        if len(wanted)==length:
            break
        elif token not in voc.keys():
            continue
        wanted.append(voc[token])
        
    while(len(wanted)<length):
        wanted.append(0)
    return np.asarray(wanted, dtype=int)
csv_data['word_num'] = csv_data['article'].apply(token2num, args=(word_voc, word_length))
csv_data['token_num'] = csv_data['word_seg'].apply(token2num, args=(token_voc, token_length))
csv_data['class'] = csv_data['class']-1

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import logging
from models import * 
logging.basicConfig(filename='simple_cnn_test.log', level=logging.INFO)
model = get_simple_cnn(len(word_voc), 256, word_length)
from keras.callbacks import *
tf_board_op = TensorBoard(log_dir='./logs',
                            write_graph=False,
                            write_images=True,
                            embeddings_freq=0, embeddings_metadata=None)
model_save_dir = './model_file/'
tf_save_op = ModelCheckpoint(model_save_dir+'{echo:02d---{val}}',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True,
                                             save_weights_only=False,
                                             mode='auto', period=5)

def get_session():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    return tf.Session(config=tf_config)
KTF.set_session(get_session())
logging.info('the train is start')
def trans_array(x, sec_dim):
    wanted = np.zeros(shape=(len(x), sec_dim))
    for i,item in enumerate(x):
        for j,it in enumerate(item):
            wanted[i][j] = it
    return np.asarray(wanted, dtype=int)
x = trans_array(csv_data['word_num'], word_length)
history = model.fit(x = x, y = csv_data['class'], 
                                     batch_size=512,
                                     epochs=100,
                                     verbose=1, 
                                     callbacks=[tf_board_op, tf_save_op],
                                     validation_split=0.2, 
                                     shuffle=True)
logging.info(history.history)
