from utils import *
import collections
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import random
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, dot, concatenate
from keras.callbacks import ModelCheckpoint
import nltk


def lstm_model(embeddingMatrix):
    q_input = Input(shape=(MAXLEN,))
    qpos_input = Input(shape=(MAXLEN,))
    t1_input = Input(shape=(MAXLEN,))
    t1pos_input = Input(shape=(MAXLEN,))
    t2_input = Input(shape=(MAXLEN,))
    t2pos_input = Input(shape=(MAXLEN,))

    embedding_layer = Embedding(embeddingMatrix.shape[0], 300, weights=[embeddingMatrix],
                                input_length=MAXLEN, trainable=True, mask_zero=True)
    pos_embedding = Embedding(40, 16, input_length=MAXLEN, trainable=True, mask_zero=True)

    q_encode = concatenate([embedding_layer(q_input), pos_embedding(qpos_input)])
    #q_encode = embedding_layer(q_input)
    q_encode = Dropout(0.2)(q_encode)
    q_encode = LSTM(64, unroll=True)(q_encode)
    q_encode = Dropout(0.2)(q_encode)

    t1_encode = concatenate([embedding_layer(t1_input), pos_embedding(t1pos_input)])
    # t1_encode = embedding_layer(t1_input)
    t1_encode = Dropout(0.2)(t1_encode)
    t1_encode = LSTM(64, unroll=True)(t1_encode)
    t1_encode = Dropout(0.2)(t1_encode)

    t2_encode = concatenate([embedding_layer(t2_input), pos_embedding(t2pos_input)])
    # t2_encode = embedding_layer(t2_input)
    t2_encode = Dropout(0.2)(t2_encode)
    t2_encode = LSTM(64, unroll=True)(t2_encode)
    t2_encode = Dropout(0.2)(t2_encode)

    t_encode = concatenate([t1_encode, t2_encode])
    t_encode = Dense(64)(t_encode)

    similarity = dot([q_encode, t_encode], axes=1, normalize=True)
    output = Dense(1, activation='sigmoid')(similarity)

    model = Model(inputs=[q_input, qpos_input, t1_input, t1pos_input, t2_input, t2pos_input], outputs=[output])
    #model = Model(inputs=[q_input, t1_input, t2_input], outputs=[output])
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.summary()
    return model


def main():
    global MAXLEN
    MAXLEN = 35
    random.seed(0)
    # np.random.seed(0)
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

    q_inp = []
    q_pos = []
    t1_inp = []
    t1_pos = []
    t2_inp = []
    t2_pos = []

    label = []

    for q in questions[:8200]:
        q_inp.append(q[0])
        q_pos.append(" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(q[0]))]))
        # q_inp.append(q[0] + " " + " ".join(q[2:6]))
        # q_pos.append(" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(q[0]+ " " + " ".join(q[2:6])))]))
        tab = q[7]
        table = tables[tab]
        t1_inp.append(table_idx[tab])
        t1_pos.append(" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(table_idx[tab]))]))
        temp = ""
        for header, cell in zip(list(table), table.iloc[0]):
            if not header.startswith('Unnamed'):
                temp += header + " "
            else:
                temp += cell + " "
        t2_inp.append(temp)
        t2_pos.append(" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(temp))]))
        label.append(1)
        neg_samp = random.sample(list(table_idx), 2)
        while neg_samp[0] == tab or neg_samp[1] == tab:
            neg_samp = random.sample(list(table_idx), 2)
        for tab in neg_samp:
            table = tables[tab]
            q_inp.append(q[0])
            q_pos.append(" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(q[0]))]))
            # q_inp.append(q[0] + " " + " ".join(q[2:6]))
            # q_pos.append(" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(q[0]+ " " + " ".join(q[2:6])))]))
            t1_inp.append(table_idx[tab])
            t1_pos.append(" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(table_idx[tab]))]))
            temp = ""
            for header, cell in zip(list(table), table.iloc[0]):
                if not header.startswith('Unnamed'):
                    temp += header + " "
                else:
                    temp += cell + " "
            t2_inp.append(temp)
            t2_pos.append(" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(temp))]))
            label.append(0)

    pos_tokenizer = Tokenizer(oov_token="unk")
    pos_tokenizer.fit_on_texts(q_pos)
    pos_tokenizer.fit_on_texts(t1_pos)
    pos_tokenizer.fit_on_texts(t2_pos)

    q_inp = tokenizer.texts_to_sequences(q_inp)
    q_inp = pad_sequences(q_inp, maxlen=MAXLEN, padding='post')

    q_pos = pos_tokenizer.texts_to_sequences(q_pos)
    q_pos = pad_sequences(q_pos, maxlen=MAXLEN, padding='post')

    t1_inp = tokenizer.texts_to_sequences(t1_inp)
    t1_inp = pad_sequences(t1_inp, maxlen=MAXLEN, padding='post')

    t1_pos = pos_tokenizer.texts_to_sequences(t1_pos)
    t1_pos = pad_sequences(t1_pos, maxlen=MAXLEN, padding='post')

    t2_inp = tokenizer.texts_to_sequences(t2_inp)
    t2_inp = pad_sequences(t2_inp, maxlen=MAXLEN, padding='post')

    t2_pos = pos_tokenizer.texts_to_sequences(t2_pos)
    t2_pos = pad_sequences(t2_pos, maxlen=MAXLEN, padding='post')

    # Fit network
    model = lstm_model(embeddingMatrix)
    checkpoint = ModelCheckpoint('tabf.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='auto')
    callbacks_list = [checkpoint]
    model.fit([q_inp, q_pos, t1_inp, t1_pos, t2_inp, t2_pos], label, validation_split=0.1, batch_size=100, epochs=40,
              callbacks=callbacks_list)
    # model.fit([q_inp, t1_inp,t2_inp], label, validation_split=0.1, batch_size=100, epochs=40,
    #           callbacks=callbacks_list)

    model = load_model('tabf.h5')

    wrong = collections.Counter()
    t1_inp = []
    t1_pos = []
    t2_inp = []
    t2_pos = []

    tab2idx = {}
    i = 0
    for tab in table_idx:
        table = tables[tab]
        t1_inp.append(table_idx[tab])
        t1_pos.append(" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(table_idx[tab]))]))
        temp = ""
        for header, cell in zip(list(table), table.iloc[0]):
            if not header.startswith('Unnamed'):
                temp += header + " "
            else:
                temp += cell + " "
        t2_inp.append(temp)
        t2_pos.append(" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(temp))]))
        tab2idx[tab] = i
        i += 1
    t1_inp = tokenizer.texts_to_sequences(t1_inp)
    t1_inp = pad_sequences(t1_inp, maxlen=MAXLEN, padding='post')

    t1_pos = pos_tokenizer.texts_to_sequences(t1_pos)
    t1_pos = pad_sequences(t1_pos, maxlen=MAXLEN, padding='post')

    t2_inp = tokenizer.texts_to_sequences(t2_inp)
    t2_inp = pad_sequences(t2_inp, maxlen=MAXLEN, padding='post')

    t2_pos = pos_tokenizer.texts_to_sequences(t2_pos)
    t2_pos = pad_sequences(t2_pos, maxlen=MAXLEN, padding='post')

    count1 = 0
    count2 = 0
    count3 = 0
    num = 0
    save_result = []
    for q in questions[8200:]:
        q_inp = [q[0]] * len(t1_inp)
        q_pos = [" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(q[0]))])] * len(t1_inp)

        # q_inp = [q[0] + " " + " ".join(q[2:6])] * len(t1_inp)
        # q_pos = [" ".join([x[1] for x in nltk.pos_tag(text_to_word_sequence(q[0]+ " " + " ".join(q[2:6])))])] * len(t1_inp)

        q_inp = tokenizer.texts_to_sequences(q_inp)
        q_inp = pad_sequences(q_inp, maxlen=MAXLEN, padding='post')
        q_pos = pos_tokenizer.texts_to_sequences(q_pos)
        q_pos = pad_sequences(q_pos, maxlen=MAXLEN, padding='post')

        pre = model.predict([q_inp, q_pos, t1_inp, t1_pos, t2_inp, t2_pos]).reshape(1, -1)
        # pre = model.predict([q_inp, t1_inp, t2_inp]).reshape(1, -1)
        predictions = pre[0].argsort()[::-1]
        save_result.append(pre[0])
        if tab2idx[q[7]] in predictions[:1]:
            count1 += 1
            count2 += 1
            count3 += 1
        # elif tab2idx[q[7]] in predictions[:2]:
        #     count2 += 0.5
        #     count3 += 0.5
        # elif tab2idx[q[7]] in predictions[:3]:
        #     count3 += 1 / 3
        else:
            print(q, q[7], list(table_idx)[predictions[0]])
            wrong[q[7]] += 1

        num += 1
    np.save('tabf', save_result)
    print(count1 / num)
    print(count2 / num)
    print(count3 / num)
    print(wrong)


if __name__ == '__main__':
    main()
