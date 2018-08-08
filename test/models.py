from keras.models import Sequential
from keras.layers import Embedding, Conv1D, Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.core import Flatten

def get_simple_cnn(vocab_size, embedding_size, length):
    model = Sequential()
    model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
        embeddings_initializer='random_normal',
            trainable=True,
        input_shape = (length,)
        ))
    model.add(Conv1D(activation="tanh",
                         filters=64,
                     kernel_size=1,
                     padding="valid"))
    model.add(Conv1D(activation="tanh",
                         filters=128,
                     kernel_size=2,
                     padding="valid"))
    model.add(Conv1D(activation="tanh",
                         filters=256,
                     kernel_size=3,
                     padding="valid"))
    model.add(Conv1D(activation="tanh",
                         filters=512,
                     kernel_size=3,
                     padding="valid"))
    model.add(Flatten())
    model.add(Dense(2048, activation='sigmoid'))
    model.add(Dense(512,  activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(19))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def get_RNN(vocab_size, embedding_size):
    model = Sequential()
    model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
        embeddings_initializer='random_normal',
            trainable=True
        ))
    model.add(LSTM(256,return_sequences=True))
    model.add(LSTM(256,dropout=0.5,return_sequences=False))
    model.add(Dense(19))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
from keras.engine import Input
from keras import backend as K
from keras.layers import Concatenate
from keras.models import Model

def mix_cnn_rnn(vocab_size, embedding_size,text_length):
    input_text = Input(shape=(text_length,), dtype='int32')
    embedding_vec = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
        embeddings_initializer='random_normal',
            trainable=True
        )(input_text)
    cnn_config=[{'kernel_size':1,'filters':64,  'padding':'same'},
                {'kernel_size':2,'filters':128,  'padding':'same'},
                {'kernel_size':3,'filters':512,  'padding':'same'},
                {'kernel_size':4,'filters':512,  'padding':'same'}]
    data_aug = []
    for i, c_conf in enumerate(cnn_config):
        data_aug.append(Conv1D(kernel_size =c_conf['kernel_size'],
                               filters = c_conf['filters'],
                               padding = c_conf['padding'],
                               name='aug_{}st'.format(i+1))(embedding_vec))
        
    concat_data = Concatenate(-1)(data_aug)
    rnn_result = LSTM(256,return_sequences=True)(concat_data)
    rnn_result = LSTM(256,dropout=0.5,return_sequences=False)(rnn_result)
    logist = Dense(19, activation='softmax')(rnn_result)
    model = Model(input=input_text, output=logist)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
