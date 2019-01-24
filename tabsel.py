from utils import *
import collections
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
import random
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, dot, Conv1D, MaxPooling1D, Flatten, \
    concatenate
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

    model = Model(inputs=[q_input, t_input], outputs=[output])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def cnn_model(embeddingMatrix):
    q_input = Input(shape=(q_inp_len,))
    t_input = Input(shape=(q_inp_len,))

    embedding_layer = Embedding(embeddingMatrix.shape[0], 300, weights=[embeddingMatrix],
                                input_length=q_inp_len, trainable=False, mask_zero=False)
    q_encode = embedding_layer(q_input)
    q_encode1 = Conv1D(filters=32, kernel_size=2, activation='tanh')(q_encode)
    q_encode1 = MaxPooling1D(pool_size=2)(q_encode1)
    q_encode1 = Flatten()(q_encode1)

    q_encode2 = Conv1D(filters=32, kernel_size=3, activation='tanh')(q_encode)
    q_encode2 = MaxPooling1D(pool_size=2)(q_encode2)
    q_encode2 = Flatten()(q_encode2)

    q_encode3 = Conv1D(filters=32, kernel_size=5, activation='tanh')(q_encode)
    q_encode3 = MaxPooling1D(pool_size=2)(q_encode3)
    q_encode3 = Flatten()(q_encode3)

    q_merged = concatenate([q_encode1, q_encode2, q_encode3])
    q_merged = Dense(32, activation="tanh")(q_merged)

    t_encode = embedding_layer(t_input)
    t_encode1 = Conv1D(filters=32, kernel_size=2, activation='tanh')(t_encode)
    t_encode1 = MaxPooling1D(pool_size=2)(t_encode1)
    t_encode1 = Flatten()(t_encode1)

    t_encode2 = Conv1D(filters=32, kernel_size=3, activation='tanh')(t_encode)
    t_encode2 = MaxPooling1D(pool_size=2)(t_encode2)
    t_encode2 = Flatten()(t_encode2)

    t_encode3 = Conv1D(filters=32, kernel_size=5, activation='tanh')(t_encode)
    t_encode3 = MaxPooling1D(pool_size=2)(t_encode3)
    t_encode3 = Flatten()(t_encode3)

    t_merged = concatenate([t_encode1, t_encode2, t_encode3])
    t_merged = Dense(32, activation="tanh")(t_merged)

    similarity = dot([q_merged, t_merged], axes=1, normalize=True)
    output = Dense(1, activation='tanh')(similarity)

    model = Model(inputs=[q_input, t_input], outputs=[output])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


questions, tables, table_idx = load_data()

acc1 = []
acc2 = []
acc3 = []

wrong = collections.Counter()

for i in range(5):
    questions = shuffle(questions, random_state=i)
    q_inp = []
    t_inp = []
    label = []

    for q in questions[:8000]:
        q_inp.append(q[0])
        t_inp.append(table_idx[q[7]] + " " + " ".join(tables[q[7]].iloc[0]))
        label.append(1)
        neg_samp = random.sample(list(table_idx), 2)
        for smp in neg_samp:
            q_inp.append(q[0])
            t_inp.append(table_idx[smp] + " " + " ".join(tables[smp].iloc[0]))
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
    wordIndex = tokenizer.word_index
    embeddingMatrix = getEmbeddingMatrix(wordIndex)

    # Fit network
    model = lstm_model(embeddingMatrix)
    checkpoint = ModelCheckpoint('lstm.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='auto')
    callbacks_list = [checkpoint]
    model.fit([q_inp, t_inp], label, validation_split=0.2, batch_size=200, epochs=20, callbacks=callbacks_list)

    model = load_model('lstm.h5')
    table_inp = []
    tab2idx = {}
    i = 0
    for t in table_idx:
        table_inp.append(table_idx[t] + " " + " ".join(tables[t].iloc[0]))
        tab2idx[t] = i
        i += 1
    tab_inp = tokenizer.texts_to_sequences(table_inp)
    tab_inp = pad_sequences(tab_inp, maxlen=q_inp_len, padding='post')
    count1 = 0
    count2 = 0
    count3 = 0
    num = 0
    for q in questions[8000:]:
        q_inp = [q[0]] * len(tab_inp)
        q_inp = tokenizer.texts_to_sequences(q_inp)
        q_inp = pad_sequences(q_inp, maxlen=q_inp_len, padding='post')
        pre = model.predict([q_inp, tab_inp]).reshape(1, -1)
        predictions = pre[0].argsort()[::-1]
        if tab2idx[q[7]] in predictions[:1]:
            count1 += 1
            count2 += 1
            count3 += 1
        elif tab2idx[q[7]] in predictions[:2]:
            count2 += 1
            count3 += 1
        elif tab2idx[q[7]] in predictions[:3]:
            count3 += 1
        else:
            wrong[q[7]] += 1
        num += 1
    print("acc1 is:", count1 / num)
    acc1.append(count1 / num)
    print("acc2 is:", count2 / num)
    acc2.append(count2 / num)
    print("acc3 is:", count3 / num)
    acc3.append(count3 / num)

print(acc1, acc2, acc3)
print('average acc', sum(acc1) / 5, sum(acc2) / 5, sum(acc3) / 5)
print('acc1 is', (max(acc1) + min(acc1)) / 2, '+-', (max(acc1) - min(acc1)) / 2)
print('acc2 is', (max(acc2) + min(acc2)) / 2, '+-', (max(acc2) - min(acc2)) / 2)
print('acc3 is', (max(acc3) + min(acc3)) / 2, '+-', (max(acc3) - min(acc3)) / 2)
print(wrong)
