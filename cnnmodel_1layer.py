# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from gensim.models import KeyedVectors
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D, Flatten
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers.recurrent import LSTM
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tflearn
import os
from collections import Counter
import timeit
import sys 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



vocab_size=5000  #5000 # max_features number of words considered in the vocabulary
maxlen=100 # max_sentence_lenght cut texts after this number of words (among top vocab_size most common words)
embedding_dim=100
nb_filter = 64
filter_length = 3
hidden_dims = 128 #256 the best
batch_s = 32  # size of the minibach (each batch will contain 32 sentences)
nb_epoch = 1
column_names = ['labels','reviews']
labels=pd.read_csv('data/shuffled_uclabel.txt', delimiter='\t', usecols=['labels'], names=column_names)
reviews=pd.read_csv('data/shuffled_uclabel.txt', delimiter='\t', usecols=['reviews'], names=column_names)

total_counts = Counter()
for _, row in reviews.iterrows():
    total_counts.update(row[0].split(' '))
print("Total words in dataset:{} ", len(total_counts))

word2index = {x[0]: i+1 for i, x in 
               enumerate(total_counts.most_common(vocab_size))}

word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}
X = np.empty((len(labels), ), dtype=list)
y = np.zeros((len(labels), ))
i=0
for _, row in reviews.iterrows():
    words=row[0].split(' ')
    wids = []
    for word in words:
        if word in word2index:
            wids.append(word2index[word])
        else:
            wids.append(word2index["UNK"])
    X[i] = wids
    i += 1
    
X = sequence.pad_sequences(X, maxlen=maxlen)
Y = np_utils.to_categorical(labels)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, 
                                                random_state=42)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
vocab_dim=len(word2index)+1

#pre-embedding
word2vec = KeyedVectors.load_word2vec_format('data/beyazperdeyorumlar_skipgram.vec', binary=False)
embedding_weights = np.zeros((vocab_dim, embedding_dim))
for word, index in word2index.items():
    try:
        embedding_weights[index, :] = word2vec[word]
    except KeyError:
        pass

def model_cnn(modelname_tosave):
    start = timeit.default_timer()
    model = Sequential()
    # CONV => RELU => POOL
    model.add(Embedding(vocab_dim, embedding_dim, input_length=maxlen))
    model.add(Conv1D(filters=nb_filter, kernel_size=filter_length))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    # Flatten => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # a softmax classifier
    model.add(Dense(2))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy",
                     metrics=["accuracy"])
    save_model='saved_models/'+modelname_tosave+'.hdf5'
    callbacks = [
    
    EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint(filepath=save_model, monitor='val_loss', save_best_only=True, verbose=1),
    ]
    history=model.fit(Xtrain, Ytrain, batch_size=batch_s,epochs=nb_epoch,
                 validation_data=(Xtest, Ytest),callbacks=callbacks, verbose=1)
    accuracy_rslt=history.history["acc"]
    val_accuracy_rslt=history.history["val_acc"]
    loss_rslt=history.history["loss"]
    val_loss_rslt=history.history["val_loss"]
    plt.subplot(211)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(history.history["acc"], color="g", label="Train")
    plt.plot(history.history["val_acc"], color="b", label="Validation")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(history.history["loss"], color="g", label="Train")
    plt.plot(history.history["val_loss"], color="g", label="Validation")
    plt.tight_layout()
    plt.savefig('results/model_cnadn1_bs64.png')
    plt.close()
    score = model.evaluate(Xtest, Ytest, verbose=1)
    print("Test score cnn: {:.3f}, accuracy: {:.3f}".format(score[0], score[1]))
    stop = timeit.default_timer()
    train_time=stop - start
    with open("results/cnn_1layer_resulasdt.txt", "a") as myfile:
        myfile.write("(cnn_model(1Layer,bs32,hl128,em100,nfilter=64,filter=3) \n accuracy={0}\n validation_accuracy={1}\n loss={2}\n validation_loss={3}\n test_score={4}, train_time={5}\n".format(
        accuracy_rslt,val_accuracy_rslt,loss_rslt,val_loss_rslt,score[1],train_time))
        
def model_cnn_pre(modelname_tosave):
    start = timeit.default_timer()
    model = Sequential()
    # CONV => RELU => POOL
    model.add(Embedding(vocab_dim, embedding_dim, weights=[embedding_weights], input_length=maxlen))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=nb_filter, kernel_size=filter_length))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    # Flatten => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))
    # a softmax classifier
    model.add(Dense(2))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy",
                     metrics=["accuracy"])
    save_model='saved_models/'+modelname_tosave+'.hdf5'
    callbacks = [
    
    EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint(filepath=save_model, monitor='val_loss', save_best_only=True, verbose=1),
    ]
    history=model.fit(Xtrain, Ytrain, batch_size=batch_s,epochs=nb_epoch,
                 validation_data=(Xtest, Ytest),callbacks=callbacks, verbose=1)
    accuracy_rslt=history.history["acc"]
    val_accuracy_rslt=history.history["val_acc"]
    loss_rslt=history.history["loss"]
    val_loss_rslt=history.history["val_loss"]
    plt.subplot(211)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(history.history["acc"], color="g", label="Train")
    plt.plot(history.history["val_acc"], color="b", label="Validation")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(history.history["loss"], color="g", label="Train")
    plt.plot(history.history["val_loss"], color="g", label="Validation")
    plt.tight_layout()
    plt.savefig('results/model_cnn1_psdre_bs64.png')
    plt.close()
    score = model.evaluate(Xtest, Ytest, verbose=1)
    print("Test score cnn: {:.3f}, accuracy: {:.3f}".format(score[0], score[1]))
    stop = timeit.default_timer()
    train_time=stop - start
    with open("results/cnn_1layader_result.txt", "a") as myfile:
        myfile.write("(cnn_model_pre(1Layer,bs32,hl128,em100,nfilter64,filter3) \n accuracy={0}\n validation_accuracy={1}\n loss={2}\n validation_loss={3}\n test_score={4}, train_time={5}\n".format(
        accuracy_rslt,val_accuracy_rslt,loss_rslt,val_loss_rslt,score[1],train_time))

model_cnn("modelcnn1_bs6sd4_32")
model_cnn_pre("modelcnn1_psdre_bs632")