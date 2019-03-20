from bm25.query import QueryProcessor
import operator
from utils import *
import nltk
import random
import collections


def main():
    USEANSWER = True
    random.seed(0)
    np.random.seed(0)

    questions, tables, table_idx = load_data()
    random.shuffle(questions)

    if USEANSWER:
        queries = [nltk.word_tokenize(q[0].lower() + " " + " ".join(q[2:6])) for q in questions[8200:]]
    else:
        queries = [nltk.word_tokenize(q[0].lower()) for q in questions[8200:]]
    reltable = [q[7] for q in questions[8200:]]
    corpus = {}

    for t in table_idx:
        table = tables[t]
        header = list(table)
        cells = table.applymap(str).values
        body = ""
        for row in cells:
            body += " ".join(row) + " "
        temp = []
        for h in header:
            if not h.startswith('Unnamed'):
                temp += nltk.word_tokenize(h)
        doc = nltk.word_tokenize(table_idx[t]) + temp + nltk.word_tokenize(body)
        corpus[t] = [tok.lower() for tok in doc]

    # wrong = collections.Counter()
    # count = collections.Counter()
    proc = QueryProcessor(queries, corpus)
    results = proc.run()
    ap1 = 0
    ap2 = 0
    ap3 = 0
    for i in range(len(results)):
        # count[reltable[i]] += 1
        if results[i]:
            sorted_x = sorted(results[i].items(), key=operator.itemgetter(1), reverse=True)
            res = [x[0] for x in sorted_x[:3]]
            if reltable[i] in res[:1]:
                ap1 += 1
                ap2 += 1
                ap3 += 1
            elif reltable[i] in res[:2]:
                ap2 += 0.5
                ap3 += 0.5
            elif reltable[i] in res:
                ap3 += 1 / 3
    print(ap1 / len(queries))
    print(ap2 / len(queries))
    print(ap3 / len(queries))


if __name__ == '__main__':
    main()
