from bm25.query import QueryProcessor
import operator
from utils import *
import nltk


def main():
    questions, tables, table_idx = load_data()

    queries = [nltk.word_tokenize(q[0]) for q in questions]
    corpus = {}

    # title + header + first row
    for t in table_idx:
        header = list(tables[t])
        temp = []
        for h in header:
            if not h.startswith('Unnamed'):
                temp += nltk.word_tokenize(h)
        corpus[t] = nltk.word_tokenize(table_idx[t]) + temp + nltk.word_tokenize(" ".join(tables[t].iloc[0]))

    proc = QueryProcessor(queries, corpus)
    results = proc.run()
    correct = 0
    for i in range(len(results)):
        if results[i]:
            sorted_x = sorted(results[i].items(), key=operator.itemgetter(1))
            res = sorted_x[-1][0]
            if res == questions[i][7]:
                correct += 1
    print(correct / len(questions))


if __name__ == '__main__':
    main()
