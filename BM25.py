from bm25.query import QueryProcessor
import operator
from utils import *
import nltk
import collections


def main():
    questions, tables, table_idx = load_data()
    test = ['regents-02', 'regents-07', 'regents-10', 'regents-17', 'regents-19', 'regents-24', 'regents-25&26',
            'regents-34', 'regents-40', 'monarch-46', 'monarch-50', 'monarch-53', 'monarch-62', 'monarch-64',
            'monarch-67']

    queries = [nltk.word_tokenize(q[0].lower() + " ".join(q[2:6]).lower()) for q in questions if q[7] in test]
    # print(queries)
    reltable = [q[7] for q in questions if q[7] in test]
    corpus = {}

    # title + header + body
    for t in test:
        table = tables[t]
        header = list(table)
        cells = table.applymap(str).values
        body = ""
        for row in cells:
            body += " ".join(row)
        temp = []
        for h in header:
            if not h.startswith('Unnamed'):
                temp += nltk.word_tokenize(h)
        doc = nltk.word_tokenize(table_idx[t]) + temp + nltk.word_tokenize(body)
        corpus[t] = [tok.lower() for tok in doc]

    wrong = collections.Counter()
    count = collections.Counter()
    proc = QueryProcessor(queries, corpus)
    results = proc.run()
    correct = 0
    for i in range(len(results)):
        count[reltable[i]] += 1
        if results[i]:
            sorted_x = sorted(results[i].items(), key=operator.itemgetter(1), reverse=True)
            res = [x[0] for x in sorted_x[:1]]
            if reltable[i] in res:
                correct += 1
            else:
                wrong[reltable[i]] += 1
    print(correct / len(queries))
    for t in test:
        print(t, wrong[t] / count[t])


if __name__ == '__main__':
    main()
