from utils import *
import random
import Levenshtein
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, dot, Conv1D, MaxPooling1D, multiply, Activation, Bidirectional
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from collections import defaultdict
from nltk.corpus import stopwords


def basic(embeddingMatrix):
    q_input = Input(shape=(MAXLEN,))
    t_input = Input(shape=(MAXLEN,))
    feat = Input(shape=(2, ))

    embedding_layer = Embedding(embeddingMatrix.shape[0], 300, weights=[embeddingMatrix],
                                input_length=MAXLEN, trainable=True, mask_zero=True)
    q_encode = embedding_layer(q_input)
    t_encode = embedding_layer(t_input)

    element_wise_dot_product = dot([t_encode, q_encode], axes=-1, normalize=True)
    element = Conv1D(filters=5, kernel_size=2, activation='tanh', padding='same')(element_wise_dot_product)
    attention = MaxPooling1D(pool_size=5, data_format='channels_first')(element)
    attention = Activation('tanh')(attention)
    t_encode = multiply([t_encode, attention])

    # q_encode = Bidirectional(LSTM(64, unroll=True))(q_encode)
    # t_encode = Bidirectional(LSTM(64, unroll=True))(t_encode)

    q_encode = LSTM(64, unroll=True)(q_encode)
    t_encode = LSTM(64, unroll=True)(t_encode)

    # similarity = dot([q_encode, t_encode], axes=1, normalize=True)
    t_transform = Dense(64, use_bias=False)(t_encode)
    similarity = dot([q_encode, t_transform], axes=1, normalize=True)
    merged = concatenate([q_encode, t_encode, similarity, feat])
    hidden = Dense(32, activation='tanh')(merged)
    output = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=[q_input, t_input, feat], outputs=[output])
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.summary()
    return model


def main():
    global MAXLEN
    MAXLEN = 35
    threshold = 0.8
    random.seed(0)
    np.random.seed(0)

    questions, tables, table_idx = load_data()
    random.shuffle(questions)

    words = []
    for q in questions:
        words.append(q[0] + " " + " ".join(q[2:6]))
    for t in table_idx:
        cells = tables[t].applymap(str).values
        body = ""
        for row in cells:
            body += " ".join(row) + " "
        words.append(body)

    print("Extracting tokens...")
    tokenizer = Tokenizer(oov_token="unk")
    tokenizer.fit_on_texts(words)
    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))
    embeddingMatrix = getEmbeddingMatrix(wordIndex)

    stop_words = set(stopwords.words('english'))
    q_inp = []
    t_inp = []
    feat = []
    label = []
    for q in questions[:8200]:
        tab = q[7]
        df = tables[tab].applymap(str)
        headers = list(df)
        rele_row = int(q[8]) - 1
        rele_col = int(q[9])
        gold = df.iloc[rele_row]
        q_inp.append(q[0])
        gold.iloc[rele_col] = ''
        gold = " ".join(gold)
        t_inp.append(gold)

        query = text_to_word_sequence(q[0])
        row = set(text_to_word_sequence(gold))
        temp1 = 0
        temp2 = 0
        for tok in query:
            if tok in row:
                if tok not in stop_words:
                    temp2 += 1
                temp1 += 1
        feat.append([temp1, temp2])
        label.append(1)
        table = df.drop_duplicates(df.columns.difference([headers[rele_col]]))
        if len(table) >= 2:
            negsmp = table.sample(2)
            negsmp.iloc[:, rele_col] = ''
            for idx, row in negsmp.iterrows():
                temp = " ".join(row)
                q_inp.append(q[0])
                t_inp.append(temp)
                if temp == gold:
                    label.append(1)
                else:
                    label.append(0)
                temp = set(text_to_word_sequence(temp))
                temp1 = 0
                temp2 = 0
                for tok in query:
                    if tok in temp:
                        if tok not in stop_words:
                            temp2 += 1
                        temp1 += 1
                feat.append([temp1, temp2])

    q_inp = tokenizer.texts_to_sequences(q_inp)
    t_inp = tokenizer.texts_to_sequences(t_inp)
    q_inp = pad_sequences(q_inp, maxlen=MAXLEN, padding='post')
    t_inp = pad_sequences(t_inp, maxlen=MAXLEN, padding='post')
    feat = np.array(feat)
    label = np.array(label)
    print("Shape of q_inp tensor: ", q_inp.shape)
    print("Shape of t_inp tensor: ", t_inp.shape)
    print("Shape of label tensor: ", label.shape)

    model = basic(embeddingMatrix)
    checkpoint = ModelCheckpoint('basic.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    model.fit([q_inp, t_inp, feat], label, epochs=15, batch_size=100, validation_split=0.1, callbacks=callbacks_list)
    model = load_model('basic.h5')

    # Test
    correct = 0
    count = 0
    for q in questions[8200:]:
        count += 1
        tab = q[7]
        df = tables[tab].applymap(str)
        rele_col = int(q[9])
        pattern = set()
        ans = defaultdict(set)
        for idx, row in df.iterrows():
            cell = row.iloc[rele_col]
            row.iloc[rele_col] = ''
            pat = " ".join(row)
            pattern.add(pat)
            ans[pat].add(cell)
        q_inp = [q[0]] * len(pattern)
        pattern = list(pattern)

        query = text_to_word_sequence(q[0])
        feat = []
        for pp in pattern:
            temp = set(text_to_word_sequence(pp))
            temp1 = 0
            temp2 = 0
            for tok in query:
                if tok in temp:
                    if tok not in stop_words:
                        temp2 += 1
                    temp1 += 1
            feat.append([temp1, temp2])

        q_inp = tokenizer.texts_to_sequences(q_inp)
        t_inp = tokenizer.texts_to_sequences(pattern)
        q_inp = pad_sequences(q_inp, maxlen=MAXLEN, padding='post')
        t_inp = pad_sequences(t_inp, maxlen=MAXLEN, padding='post')
        feat = np.array(feat)
        pre = model.predict([q_inp, t_inp, feat]).reshape(1, -1)
        predictions = pre[0].argsort()[::-1]

        choices = [x.lower().strip() for x in q[2:6]]
        result = -1
        for i in predictions:
            score = []
            for c in range(4):
                score.append(max(Levenshtein.ratio(choices[c], cell) for cell in list(ans[pattern[i]])))
            if max(score) > threshold:
                result = np.argmax(np.array(score))
                break
        # if result == -1:
        #     result = random.randint(0, 3)
        if result == -1:
            score = []
            for c in range(4):
                score.append(max(Levenshtein.ratio(choices[c], cell) for cell in list(ans[pattern[predictions[0]]])))
            result = np.argmax(np.array(score))
        if result == int(q[6]) - 1:
            correct += 1
    print(correct/count)


if __name__ == '__main__':
    main()
