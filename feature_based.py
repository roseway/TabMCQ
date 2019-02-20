from utils import *
from bm25.query import QueryProcessor
import random
from keras.layers import Dense
from keras.models import Sequential
import collections
import nltk
from difflib import SequenceMatcher

def main():
    questions, tables, table_idx = load_data()

    test = ['regents-02', 'regents-03', 'regents-08', 'regents-13', 'regents-17', 'regents-19', 'regents-22',
            'regents-25&26', 'regents-42', 'monarch-44', 'monarch-47', 'monarch-50', 'monarch-53', 'monarch-57',
            'monarch-62', 'monarch-64']
    train = [t for t in table_idx if t not in test]

    trainx, trainy = [], []
    train_questions = [q for q in questions if q[7] in train]
    test_questions = [q for q in questions if q[7] in test]
    cap_corpus = {}
    header_corpus = {}
    cell_corpus = {}

    for t in train:
        table = tables[t]
        cap_corpus[t] = nltk.word_tokenize(table_idx[t].lower())
        header = list(table)
        temp = []
        for h in header:
            if not h.startswith('Unnamed'):
                temp += nltk.word_tokenize(h.lower())
        header_corpus[t] = temp
        cells = table.applymap(str).values
        body = ""
        for row in cells:
            body += " ".join(row) + " "
        cell_corpus[t] = nltk.word_tokenize(body.lower())

    train_queries = [nltk.word_tokenize(q[0].lower()) for q in train_questions]

    cap_bm25 = QueryProcessor(train_queries, cap_corpus)
    cap_results = cap_bm25.run()
    cap_idf = cap_bm25.idf()

    header_bm25 = QueryProcessor(train_queries, header_corpus)
    header_results = header_bm25.run()
    header_idf = header_bm25.idf()

    cell_bm25 = QueryProcessor(train_queries, cell_corpus)
    cell_results = cell_bm25.run()
    cell_idf = cell_bm25.idf()

    for i in range(len(train_questions)):
        q = train_questions[i]
        q_tok = nltk.word_tokenize(q[0].lower())
        x = []
        tab = q[7]
        table = tables[tab]
        cap = table_idx[tab].lower()
        temp = [h.lower() for h in list(table) if not h.startswith('Unnamed')]
        header = ""
        for h in temp:
            header += h
        cell = table.applymap(str).values
        body = ""
        for row in cell:
            body += " ".join(row) + " "
        body = body.lower()
        capc = collections.Counter(nltk.word_tokenize(cap))
        headerc = collections.Counter(nltk.word_tokenize(header))
        bodyc = collections.Counter(nltk.word_tokenize(body))
        que = " ".join(q_tok)
        cap = " ".join(capc)
        header = " ".join(headerc)
        body = " ".join(bodyc)

        x.append(len(q_tok))  # Query length

        x.append(cap_idf[i])  # Sum of query idf score
        x.append(header_idf[i])  # Sum of query idf score
        x.append(cell_idf[i])  # Sum of query idf score

        x.append(SequenceMatcher(None, que, cap).find_longest_match(0, len(que), 0, len(cap)).size/len(que))
        x.append(SequenceMatcher(None, que, header).find_longest_match(0, len(que), 0, len(header)).size/len(que))
        x.append(SequenceMatcher(None, que, body).find_longest_match(0, len(que), 0, len(body)).size/len(que))

        x.append(sum([capc[tok]/sum(capc.values()) for tok in q_tok]))  # Term frequency in cap
        x.append(sum([headerc[tok]/sum(headerc.values()) for tok in q_tok]))  # Term frequency in header
        x.append(sum([bodyc[tok]/sum(bodyc.values()) for tok in q_tok]))  # Term frequency in body

        x.append(max([capc[tok] / sum(capc.values()) for tok in q_tok]))  # Term frequency in cap
        x.append(max([headerc[tok] / sum(headerc.values()) for tok in q_tok]))  # Term frequency in header
        x.append(max([bodyc[tok] / sum(bodyc.values()) for tok in q_tok]))  # Term frequency in body

        x.append(sum([capc[tok]/sum(capc.values()) for tok in q_tok])/len(q_tok))  # Term frequency in cap
        x.append(sum([headerc[tok]/sum(headerc.values()) for tok in q_tok])/len(q_tok)) # Term frequency in header
        x.append(sum([bodyc[tok]/sum(bodyc.values()) for tok in q_tok])/len(q_tok))  # Term frequency in body

        x.append(cap_results[i][tab])
        x.append(header_results[i][tab])
        x.append(cell_results[i][tab])
        trainx.append(x)
        trainy.append(1)
        neg_samp = random.sample(train, 2)
        while neg_samp[0] == tab or neg_samp[1] == tab:
            neg_samp = random.sample(train, 2)
        for tab in neg_samp:
            x = []
            table = tables[tab]
            cap = table_idx[tab].lower()
            temp = [h.lower() for h in list(table) if not h.startswith('Unnamed')]
            header = ""
            for h in temp:
                header += h
            cell = table.applymap(str).values
            body = ""
            for row in cell:
                body += " ".join(row) + " "
            body = body.lower()
            capc = collections.Counter(nltk.word_tokenize(cap))
            headerc = collections.Counter(nltk.word_tokenize(header))
            bodyc = collections.Counter(nltk.word_tokenize(body))
            cap = " ".join(capc)
            header = " ".join(headerc)
            body = " ".join(bodyc)

            x.append(len(q_tok))  # Query length

            x.append(cap_idf[i])  # Sum of query idf score
            x.append(header_idf[i])  # Sum of query idf score
            x.append(cell_idf[i])  # Sum of query idf score

            x.append(SequenceMatcher(None, que, cap).find_longest_match(0, len(que), 0, len(cap)).size/len(que))
            x.append(SequenceMatcher(None, que, header).find_longest_match(0, len(que), 0, len(header)).size/len(que))
            x.append(SequenceMatcher(None, que, body).find_longest_match(0, len(que), 0, len(body)).size/len(que))

            x.append(sum([capc[tok] / sum(capc.values()) for tok in q_tok]))  # Term frequency in cap
            x.append(sum([headerc[tok] / sum(headerc.values()) for tok in q_tok]))  # Term frequency in header
            x.append(sum([bodyc[tok] / sum(bodyc.values()) for tok in q_tok]))  # Term frequency in body

            x.append(max([capc[tok] / sum(capc.values()) for tok in q_tok]))  # Term frequency in cap
            x.append(max([headerc[tok] / sum(headerc.values()) for tok in q_tok]))  # Term frequency in header
            x.append(max([bodyc[tok] / sum(bodyc.values()) for tok in q_tok]))  # Term frequency in body

            x.append(sum([capc[tok] / sum(capc.values()) for tok in q_tok]) / len(q_tok))  # Term frequency in cap
            x.append(sum([headerc[tok] / sum(headerc.values()) for tok in q_tok]) / len(q_tok))  # Term frequency in header
            x.append(sum([bodyc[tok] / sum(bodyc.values()) for tok in q_tok]) / len(q_tok))  # Term frequency in body

            x.append(cap_results[i][tab])
            x.append(header_results[i][tab])
            x.append(cell_results[i][tab])

            trainx.append(x)
            trainy.append(0)
    inplen = len(trainx[0])
    trainx = np.array(trainx)
    trainy = np.array(trainy)
    print(trainx[:3])
    print(trainy[:3])

    model = Sequential()
    model.add(Dense(16, input_shape=(inplen,), activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    model.fit(trainx, trainy, validation_split=0.2, batch_size=200, epochs=50)

    cap_corpus = {}
    header_corpus = {}
    cell_corpus = {}

    # BM25 scores
    for t in test:
        table = tables[t]
        cap_corpus[t] = nltk.word_tokenize(table_idx[t].lower())
        header = list(table)
        temp = []
        for h in header:
            if not h.startswith('Unnamed'):
                temp += nltk.word_tokenize(h.lower())
        header_corpus[t] = temp
        cells = table.applymap(str).values
        body = ""
        for row in cells:
            body += " ".join(row) + " "
        cell_corpus[t] = nltk.word_tokenize(body.lower())

    test_queries = [nltk.word_tokenize(q[0].lower()) for q in test_questions]

    cap_bm25 = QueryProcessor(test_queries, cap_corpus)
    cap_results = cap_bm25.run()
    cap_idf = cap_bm25.idf()

    header_bm25 = QueryProcessor(test_queries, header_corpus)
    header_results = header_bm25.run()
    header_idf = header_bm25.idf()

    cell_bm25 = QueryProcessor(test_queries, cell_corpus)
    cell_results = cell_bm25.run()
    cell_idf = cell_bm25.idf()

    correct = 0
    wrong = collections.Counter()
    count = collections.Counter()
    for i in range(len(test_questions)):
        q = test_questions[i]
        q_tok = nltk.word_tokenize(q[0].lower())
        count[q[7]] += 1
        testx = []
        for tab in test:
            x = []
            table = tables[tab]
            cap = table_idx[tab].lower()
            temp = [h.lower() for h in list(table) if not h.startswith('Unnamed')]
            header = ""
            for h in temp:
                header += h
            cell = table.applymap(str).values
            body = ""
            for row in cell:
                body += " ".join(row) + " "
            body = body.lower()
            capc = collections.Counter(nltk.word_tokenize(cap))
            headerc = collections.Counter(nltk.word_tokenize(header))
            bodyc = collections.Counter(nltk.word_tokenize(body))
            que = " ".join(q_tok)
            cap = " ".join(capc)
            header = " ".join(headerc)
            body = " ".join(bodyc)

            x.append(len(q_tok))  # Query length

            x.append(cap_idf[i])  # Sum of query idf score
            x.append(header_idf[i])  # Sum of query idf score
            x.append(cell_idf[i])  # Sum of query idf score

            x.append(SequenceMatcher(None, que, cap).find_longest_match(0, len(que), 0, len(cap)).size/len(que))
            x.append(SequenceMatcher(None, que, header).find_longest_match(0, len(que), 0, len(header)).size/len(que))
            x.append(SequenceMatcher(None, que, body).find_longest_match(0, len(que), 0, len(body)).size/len(que))

            x.append(sum([capc[tok] / sum(capc.values()) for tok in q_tok]))  # Term frequency in cap
            x.append(sum([headerc[tok] / sum(headerc.values()) for tok in q_tok]))  # Term frequency in header
            x.append(sum([bodyc[tok] / sum(bodyc.values()) for tok in q_tok]))  # Term frequency in body

            x.append(max([capc[tok] / sum(capc.values()) for tok in q_tok]))  # Term frequency in cap
            x.append(max([headerc[tok] / sum(headerc.values()) for tok in q_tok]))  # Term frequency in header
            x.append(max([bodyc[tok] / sum(bodyc.values()) for tok in q_tok]))  # Term frequency in body

            x.append(sum([capc[tok] / sum(capc.values()) for tok in q_tok]) / len(q_tok))  # Term frequency in cap
            x.append(sum([headerc[tok] / sum(headerc.values()) for tok in q_tok]) / len(q_tok))  # Term frequency in header
            x.append(sum([bodyc[tok] / sum(bodyc.values()) for tok in q_tok]) / len(q_tok))  # Term frequency in body

            x.append(cap_results[i][tab])
            x.append(header_results[i][tab])
            x.append(cell_results[i][tab])
            testx.append(x)
        testx = np.array(testx)
        pre = model.predict(testx).reshape(1, -1)
        predictions = pre[0].argsort()[::-1]
        if test[predictions[0]] == q[7]:
            correct += 1
        else:
            wrong[q[7]] += 1
    print(correct / len(test_questions))
    for t in test:
        print(t, wrong[t] / count[t])


if __name__ == '__main__':
    main()
