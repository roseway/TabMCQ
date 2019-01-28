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


def selcol(embeddingMatrix):
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
    output = Dense(1, activation='sigmoid')(similarity)

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
    output = Dense(1, activation='sigmoid')(similarity)

    model = Model(inputs=[q_input, t_input], outputs=[output])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


questions, tables, table_idx = load_data()

acc = []
wrong = collections.Counter()
epc = 5
for i in range(epc):
    questions = shuffle(questions, random_state=i)
    q_inp = []
    t_inp = []
    label = []
    count = 0
    for q in questions[:8000]:
        table = tables[q[7]]
        for i, c in enumerate(table):
            if not table.columns[int(q[9])].startswith('Unnamed'):
                q_inp.append(" ".join(q[2:6]))
                t_inp.append(" ".join(str(table[c][:4])))
                if i == int(q[9]):
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
    print("Found %s unique tokens." % len(wordIndex))
    embeddingMatrix = getEmbeddingMatrix(wordIndex)

    # Fit network
    model = selcol(embeddingMatrix)
    checkpoint = ModelCheckpoint('selcol.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='auto')
    callbacks_list = [checkpoint]
    model.fit([q_inp, t_inp], label, validation_split=0.2, batch_size=200, epochs=20, callbacks=callbacks_list)

    model = load_model('selcol.h5')

    count = 0
    num = 0
    for q in questions[8000:]:
        table = tables[q[7]]
        named = []
        q_inp = []
        t_inp = []
        for i, c in enumerate(table):
            if not table.columns[int(q[9])].startswith('Unnamed'):
                named.append(i)
                q_inp.append(" ".join(q[2:6]))
                t_inp.append(" ".join(str(table[c][:4])))

        q_inp = tokenizer.texts_to_sequences(q_inp)
        q_inp = pad_sequences(q_inp, maxlen=q_inp_len, padding='post')
        t_inp = tokenizer.texts_to_sequences(t_inp)
        t_inp = pad_sequences(t_inp, maxlen=q_inp_len, padding='post')
        pre = model.predict([q_inp, t_inp]).reshape(1, -1)
        prediction = pre[0].argmax()
        if named[prediction] == int(q[9]):
            count += 1
        else:
            wrong[q[7]] += 1
        num += 1
    print("acc is:", count / num)
    acc.append(count / num)

print(acc)
print('average acc', sum(acc) / epc)
print('acc1 is', (max(acc) + min(acc)) / 2, '+-', (max(acc) - min(acc)) / 2)
print(wrong)
