from bm25.query import QueryProcessor
import operator
from utils import *
import nltk
import collections


def main():
    questions, tables, table_idx = load_data()
    test = ['regents-05&09', 'regents-07', 'regents-08', 'regents-10', 'regents-24', 'regents-31',
            'regents-37', 'monarch-44', 'monarch-49', 'monarch-50', 'monarch-56', 'monarch-63']

    queries = [nltk.word_tokenize(q[0].lower()) for q in questions if q[7] in test]
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
            sorted_x = sorted(results[i].items(), key=operator.itemgetter(1))
            res = sorted_x[-1][0]
            if res == reltable[i]:
                correct += 1
            else:
                wrong[reltable[i]] += 1
    print(correct / len(queries))
    # for t in test:
    #     print(t, wrong[t]/count[t])


if __name__ == '__main__':
    main()
