from utils import *
from bm25.query import QueryProcessor
import random
from keras.layers import Dense
from keras.models import Sequential
import collections
import nltk
from difflib import SequenceMatcher
from statistics import mean
import Levenshtein
import string


def main():
    USEANSWER = True
    random.seed(5)
    questions, tables, table_idx = load_data()
    test = ['regents-02', 'regents-03', 'regents-08', 'regents-13', 'regents-17', 'regents-19', 'regents-22',
            'regents-25&26', 'regents-42', 'monarch-44', 'monarch-47', 'monarch-50', 'monarch-53', 'monarch-57',
            'monarch-62', 'monarch-64']
    train = [t for t in table_idx if t not in test]
    punc_table = str.maketrans({key: None for key in string.punctuation})

    trainx, trainy = [], []
    train_questions = [q for q in questions if q[7] in train]
    test_questions = [q for q in questions if q[7] in test]

    cap_corpus = {}
    header_corpus = {}
    cell_corpus = {}
    for t in train:
        table = tables[t]
        cap_corpus[t] = nltk.word_tokenize(table_idx[t].translate(punc_table).lower())
        header = list(table)
        temp = []
        for h in header:
            if not h.startswith('Unnamed'):
                temp += nltk.word_tokenize(h.translate(punc_table).lower())
        header_corpus[t] = temp
        cells = table.applymap(str).values
        body = ""
        for row in cells:
            body += " ".join(row) + " "
        cell_corpus[t] = nltk.word_tokenize(body.translate(punc_table).lower())

    vocab = []
    for t in train:
        vocab += cap_corpus[t]
        vocab += header_corpus[t]
        vocab += cell_corpus[t]
    vocab = set(vocab)

    if USEANSWER:
        train_queries = [nltk.word_tokenize(
            q[0].translate(punc_table).lower() + " " + " ".join(q[2:6]).translate(punc_table).lower()) for q in
                         train_questions]
    else:
        train_queries = [nltk.word_tokenize(q[0].translate(punc_table).lower()) for q in train_questions]

    # Compute bm25 scores and idf scores
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
        q_tok = train_queries[i]
        tab = q[7]
        table = tables[tab]

        cap = cap_corpus[tab]
        header = header_corpus[tab]
        body = cell_corpus[tab]

        capc = collections.Counter(cap)
        headerc = collections.Counter(header)
        bodyc = collections.Counter(body)

        x = list()
        # Query length
        x.append(len(q_tok))

        # Sum of idf scores
        x.append(sum(cap_idf[i]))
        x.append(sum(header_idf[i]))
        x.append(sum(cell_idf[i]))

        # Max of idf scores
        x.append(max(cap_idf[i]))
        x.append(max(header_idf[i]))
        x.append(max(cell_idf[i]))

        # Average of idf scores
        x.append(mean(cap_idf[i]))
        x.append(mean(header_idf[i]))
        x.append(mean(cell_idf[i]))

        # Num of columns
        x.append(len(list(table)))

        # LCS normalized by length of query
        que = " ".join(q_tok)
        cap = " ".join(cap)
        header = " ".join(header)
        body = " ".join(body)

        x.append(SequenceMatcher(None, que, cap).find_longest_match(0, len(que), 0, len(cap)).size / len(que))
        x.append(SequenceMatcher(None, que, header).find_longest_match(0, len(que), 0, len(header)).size / len(que))
        x.append(SequenceMatcher(None, que, body).find_longest_match(0, len(que), 0, len(body)).size / len(que))

        # Term frequency
        cap_tf = [capc[tok] / sum(capc.values()) for tok in q_tok]
        header_tf = [headerc[tok] / sum(headerc.values()) for tok in q_tok]
        body_tf = [bodyc[tok] / sum(bodyc.values()) for tok in q_tok]

        # Sum of term frequency
        x.append(sum(cap_tf))
        x.append(sum(header_tf))
        x.append(sum(body_tf))

        # Max of term frequency
        x.append(max(cap_tf))
        x.append(max(header_tf))
        x.append(max(body_tf))

        # Average of term frequency
        x.append(mean(cap_tf))
        x.append(mean(header_tf))
        x.append(mean(body_tf))

        # BM25 scores
        x.append(cap_results[i][tab])
        x.append(header_results[i][tab])
        x.append(cell_results[i][tab])

        # Fix typo
        cap_typo = []
        header_typo = []
        cell_typo = []
        for tok in q_tok:
            if tok not in vocab:
                cap_typo.append(max(Levenshtein.ratio(tok, cc) for cc in capc))
                header_typo.append(max(Levenshtein.ratio(tok, cc) for cc in headerc))
                cell_typo.append(max(Levenshtein.ratio(tok, cc) for cc in bodyc))

        if not cap_typo:
            x += [0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            x.append(sum(cap_typo))
            x.append(sum(header_typo))
            x.append(sum(cell_typo))

            x.append(max(cap_typo))
            x.append(max(header_typo))
            x.append(max(cell_typo))

            x.append(mean(cap_typo))
            x.append(mean(header_typo))
            x.append(mean(cell_typo))

        trainx.append(x)
        trainy.append(1)

        # Negative samples
        neg_samp = random.sample(train, 2)
        while neg_samp[0] == tab or neg_samp[1] == tab:
            neg_samp = random.sample(train, 2)
        for tab in neg_samp:
            table = tables[tab]
            cap = cap_corpus[tab]
            header = header_corpus[tab]
            body = cell_corpus[tab]

            capc = collections.Counter(cap)
            headerc = collections.Counter(header)
            bodyc = collections.Counter(body)

            x = list()
            # Query length
            x.append(len(q_tok))

            # Sum of idf scores
            x.append(sum(cap_idf[i]))
            x.append(sum(header_idf[i]))
            x.append(sum(cell_idf[i]))

            # Max of idf scores
            x.append(max(cap_idf[i]))
            x.append(max(header_idf[i]))
            x.append(max(cell_idf[i]))

            # Average of idf scores
            x.append(mean(cap_idf[i]))
            x.append(mean(header_idf[i]))
            x.append(mean(cell_idf[i]))

            # Num of columns
            x.append(len(list(table)))

            # LCS normalized by length of query
            que = " ".join(q_tok)
            cap = " ".join(cap)
            header = " ".join(header)
            body = " ".join(body)

            x.append(SequenceMatcher(None, que, cap).find_longest_match(0, len(que), 0, len(cap)).size / len(que))
            x.append(SequenceMatcher(None, que, header).find_longest_match(0, len(que), 0, len(header)).size / len(que))
            x.append(SequenceMatcher(None, que, body).find_longest_match(0, len(que), 0, len(body)).size / len(que))

            # Term frequency
            cap_tf = [capc[tok] / sum(capc.values()) for tok in q_tok]
            header_tf = [headerc[tok] / sum(headerc.values()) for tok in q_tok]
            body_tf = [bodyc[tok] / sum(bodyc.values()) for tok in q_tok]

            # Sum of term frequency
            x.append(sum(cap_tf))
            x.append(sum(header_tf))
            x.append(sum(body_tf))

            # Max of term frequency
            x.append(max(cap_tf))
            x.append(max(header_tf))
            x.append(max(body_tf))

            # Average of term frequency
            x.append(mean(cap_tf))
            x.append(mean(header_tf))
            x.append(mean(body_tf))

            # BM25 scores
            x.append(cap_results[i][tab])
            x.append(header_results[i][tab])
            x.append(cell_results[i][tab])

            # Fix typo
            cap_typo = []
            header_typo = []
            cell_typo = []
            for tok in q_tok:
                if tok not in vocab:
                    cap_typo.append(max(Levenshtein.ratio(tok, cc) for cc in capc))
                    header_typo.append(max(Levenshtein.ratio(tok, cc) for cc in headerc))
                    cell_typo.append(max(Levenshtein.ratio(tok, cc) for cc in bodyc))

            if not cap_typo:
                x += [0, 0, 0, 0, 0, 0, 0, 0, 0]
            else:
                x.append(sum(cap_typo))
                x.append(sum(header_typo))
                x.append(sum(cell_typo))

                x.append(max(cap_typo))
                x.append(max(header_typo))
                x.append(max(cell_typo))

                x.append(mean(cap_typo))
                x.append(mean(header_typo))
                x.append(mean(cell_typo))

            trainx.append(x)
            trainy.append(0)

    # Train
    inplen = len(trainx[0])
    trainx = np.array(trainx)
    trainy = np.array(trainy)
    print(trainx[:3])
    print(trainy[:3])

    model = Sequential()
    model.add(Dense(32, input_shape=(inplen,), activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    model.fit(trainx, trainy, batch_size=200, epochs=40)

    # Test
    cap_corpus = {}
    header_corpus = {}
    cell_corpus = {}
    for t in test:
        table = tables[t]
        cap_corpus[t] = nltk.word_tokenize(table_idx[t].translate(punc_table).lower())
        header = list(table)
        temp = []
        for h in header:
            if not h.startswith('Unnamed'):
                temp += nltk.word_tokenize(h.translate(punc_table).lower())
        header_corpus[t] = temp
        cells = table.applymap(str).values
        body = ""
        for row in cells:
            body += " ".join(row) + " "
        cell_corpus[t] = nltk.word_tokenize(body.translate(punc_table).lower())

    vocab = []
    for t in test:
        vocab += cap_corpus[t]
        vocab += header_corpus[t]
        vocab += cell_corpus[t]
    vocab = set(vocab)

    if USEANSWER:
        test_queries = [nltk.word_tokenize(
            q[0].translate(punc_table).lower() + " " + " ".join(q[2:6]).translate(punc_table).lower()) for q in
                        test_questions]
    else:
        test_queries = [nltk.word_tokenize(q[0].translate(punc_table).lower()) for q in test_questions]

    cap_bm25 = QueryProcessor(test_queries, cap_corpus)
    cap_results = cap_bm25.run()
    cap_idf = cap_bm25.idf()

    header_bm25 = QueryProcessor(test_queries, header_corpus)
    header_results = header_bm25.run()
    header_idf = header_bm25.idf()

    cell_bm25 = QueryProcessor(test_queries, cell_corpus)
    cell_results = cell_bm25.run()
    cell_idf = cell_bm25.idf()

    ap1 = 0
    ap2 = 0
    ap3 = 0
    wrong = collections.Counter()
    count = collections.Counter()
    for i in range(len(test_questions)):
        q = test_questions[i]
        q_tok = test_queries[i]
        count[q[7]] += 1
        testx = []
        for tab in test:
            table = tables[tab]
            cap = cap_corpus[tab]
            header = header_corpus[tab]
            body = cell_corpus[tab]

            capc = collections.Counter(cap)
            headerc = collections.Counter(header)
            bodyc = collections.Counter(body)

            x = list()
            # Query length
            x.append(len(q_tok))

            # Sum of idf scores
            x.append(sum(cap_idf[i]))
            x.append(sum(header_idf[i]))
            x.append(sum(cell_idf[i]))

            # Max of idf scores
            x.append(max(cap_idf[i]))
            x.append(max(header_idf[i]))
            x.append(max(cell_idf[i]))

            # Average of idf scores
            x.append(mean(cap_idf[i]))
            x.append(mean(header_idf[i]))
            x.append(mean(cell_idf[i]))

            # Num of columns
            x.append(len(list(table)))

            # LCS normalized by length of query
            que = " ".join(q_tok)
            cap = " ".join(cap)
            header = " ".join(header)
            body = " ".join(body)

            x.append(SequenceMatcher(None, que, cap).find_longest_match(0, len(que), 0, len(cap)).size / len(que))
            x.append(SequenceMatcher(None, que, header).find_longest_match(0, len(que), 0, len(header)).size / len(que))
            x.append(SequenceMatcher(None, que, body).find_longest_match(0, len(que), 0, len(body)).size / len(que))

            # Term frequency
            cap_tf = [capc[tok] / sum(capc.values()) for tok in q_tok]
            header_tf = [headerc[tok] / sum(headerc.values()) for tok in q_tok]
            body_tf = [bodyc[tok] / sum(bodyc.values()) for tok in q_tok]

            # Sum of term frequency
            x.append(sum(cap_tf))
            x.append(sum(header_tf))
            x.append(sum(body_tf))

            # Max of term frequency
            x.append(max(cap_tf))
            x.append(max(header_tf))
            x.append(max(body_tf))

            # Average of term frequency
            x.append(mean(cap_tf))
            x.append(mean(header_tf))
            x.append(mean(body_tf))

            # BM25 scores
            x.append(cap_results[i][tab])
            x.append(header_results[i][tab])
            x.append(cell_results[i][tab])

            # Fix typo
            cap_typo = []
            header_typo = []
            cell_typo = []
            for tok in q_tok:
                if tok not in vocab:
                    cap_typo.append(max(Levenshtein.ratio(tok, cc) for cc in capc))
                    header_typo.append(max(Levenshtein.ratio(tok, cc) for cc in headerc))
                    cell_typo.append(max(Levenshtein.ratio(tok, cc) for cc in bodyc))

            if not cap_typo:
                x += [0, 0, 0, 0, 0, 0, 0, 0, 0]
            else:
                x.append(sum(cap_typo))
                x.append(sum(header_typo))
                x.append(sum(cell_typo))

                x.append(max(cap_typo))
                x.append(max(header_typo))
                x.append(max(cell_typo))

                x.append(mean(cap_typo))
                x.append(mean(header_typo))
                x.append(mean(cell_typo))

            testx.append(x)
        testx = np.array(testx)
        pre = model.predict(testx).reshape(1, -1)
        predictions = pre[0].argsort()[::-1]
        if test[predictions[0]] == q[7]:
            ap1 += 1
            ap2 += 1
            ap3 += 1
        # elif test[predictions[1]] == q[7]:
        #     ap2 += 0.5
        #     ap3 += 0.5
        # elif test[predictions[2]] == q[7]:
        #     ap3 += 1/3
        else:
            print(q[0])
            print(q[2:6])
            print("Pre:", test[predictions[0]], ", gold:", q[7])
    print(ap1 / len(test_questions))
    print(ap2 / len(test_questions))
    print(ap3 / len(test_questions))


if __name__ == '__main__':
    main()
