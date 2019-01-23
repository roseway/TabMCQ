from utils import *
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from math import *
import collections
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
import random
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, dot
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

def lstm_model(embeddingMatrix):
    q_input = Input(shape=(q_inp_len,))
    t_input = Input(shape=(q_inp_len,))

    embedding_layer = Embedding(embeddingMatrix.shape[0], 300, weights=[embeddingMatrix],
                                input_length=q_inp_len, trainable=False, mask_zero=True)
    q_encode = embedding_layer(q_input)
    q_encode = LSTM(64, unroll=True)(q_encode)
    q_encode = Dense(32, activation="tanh")(q_encode)

    t_encode = embedding_layer(t_input)
    t_encode = LSTM(64, unroll=True)(t_encode)
    t_encode = Dense(32, activation="tanh")(t_encode)

    similarity = dot([q_encode, t_encode], axes=1, normalize=True)
    output = Dense(1, activation='tanh')(similarity)

#    attention = Activation('softmax')(attention)
#    context = dot([attention, encoder], axes=[2, 1])
#    decoder_combined_context = concatenate([context, decoder])
#    output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)
#    output = TimeDistributed(Dense(embeddingMatrix.shape[0], activation="softmax"))(output)

    model = Model(inputs=[q_input, t_input], outputs=[output])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


questions, tables, table_idx = load_data()
questions = shuffle(questions)

splitat = 20000
q_inp = []
t_inp = []
label = []

for q in questions[:8000]:
    q_inp.append(q[0])
    t_inp.append(table_idx[q[7]])
    label.append(1)
    neg_samp = random.sample(list(table_idx), 2)
    for smp in neg_samp:
        q_inp.append(q[0])
        t_inp.append(table_idx[smp])
        if smp == q[7]:
            label.append(1)
        else:
            label.append(0)


tokenizer = Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(q_inp)
tokenizer.fit_on_texts(t_inp)
q_inp = tokenizer.texts_to_sequences(q_inp)
q_inp_len = max([len(s) for s in q_inp])
q_inp = pad_sequences(q_inp, maxlen=q_inp_len, padding='post')

t_inp = tokenizer.texts_to_sequences(t_inp)
t_inp_len = max([len(s) for s in t_inp])
t_inp = pad_sequences(t_inp, maxlen=q_inp_len, padding='post')

#label = to_categorical(np.array(label), num_classes=2)

print("Populating embedding matrix...")
wordIndex = tokenizer.word_index
embeddingMatrix = getEmbeddingMatrix(wordIndex)

# Fit network
model = lstm_model(embeddingMatrix)
# checkpoint = ModelCheckpoint('tabsel.h5', monitor='val_acc', verbose=1, save_best_only=True,
#                              mode='max')
# callbacks_list = [checkpoint]
# Fit the model
model.fit([q_inp, t_inp], label, validation_split=0.2, batch_size=200, epochs=20)

t_inp = []
idx2tab = {}
i = 0
for t in table_idx:
    t_inp.append(table_idx[t])
    idx2tab[i] = t
    i += 1

t_inp = tokenizer.texts_to_sequences(t_inp)
t_inp = pad_sequences(t_inp, maxlen=q_inp_len, padding='post')
count = 0
num = 0
for q in questions[8000:]:
    q_inp = [q[0]] * len(t_inp)
    q_inp = tokenizer.texts_to_sequences(q_inp)
    q_inp = pad_sequences(q_inp, maxlen=q_inp_len, padding='post')
    pre = model.predict([q_inp, t_inp])
    prediction = np.argmax(pre)
    if idx2tab[prediction] == q[7]:
        count += 1
    num += 1

print(count/num)
#c = ypre.shape[0]
#n = 0
#for i in range(len(xt)):
#    if ypre[i] != ytest[i].argmax():
#        n+=1
#        print(xt[i], agg[ypre[i]], agg[ytest[i].argmax()])
#print(n/c)
#
#count1 = 0
#count2 = 0
#count3 = 0

#rec = collections.Counter()
#for i in range(len(questions)):
#    scores = np.array([cosine_similarity(q_emb[i], t_emb) for t_emb in row_embeddings])
#    x = scores.argsort()[::-1]
#    if tab2idx[questions[i][7]] in x[:1]:
#        count1 += 1
#        count2 += 1
#        count3 += 1
#    elif tab2idx[questions[i][7]] in x[:2]:
#        count2 += 1
#        count3 += 1
#    elif tab2idx[questions[i][7]] in x[:3]:
#        count3 += 1
#    else:
#        rec[questions[i][7]] += 1

#print(count1 / len(questions))
#print(count2 / len(questions))
#print(count3 / len(questions))

#print(rec)
