import csv
from os import listdir
import pandas as pd
import numpy as np
from math import sqrt
import nltk


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / denominator, 3)


def load_data():
    question_path = "data/MCQs.tsv"
    table_path = ["data/tables/regents", "data/tables/monarch"]
    tableidx_path = "data/tableindex.txt"
    question_data = []
    table_data = {}
    table_index = {}
    with open(question_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            question_data.append(line)

    for path in table_path:
        name = path.split('/')[-1]
        for f in listdir(path):
            table = pd.read_csv(path + '/' + f, sep="\t")
            table_data[name + "-" + f.split(".")[0]] = table

    with open(tableidx_path) as f:
        for line in f:
            line = line.strip().split('\t')
            table_index[line[0]] = line[1]

    return question_data[1:], table_data, table_index


def getEmbeddingMatrix(wordIndex):
    """
    Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokenizer
    Output:
        embeddingMatrix : A matrix where every row has 300 dimensional GloVe embedding
    """
    print("Populating embedding matrix...")
    embeddingsIndex = {}
    # Load the embedding vectors from the GloVe file
    with open('data/glove.6B.300d.txt', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, 300))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
    return embeddingMatrix


def jaccard_ngram(string1, string2):
    ng1_chars = set(nltk.ngrams(string1, n=3, pad_left=True, pad_right=True, left_pad_symbol=' ', right_pad_symbol=' '))
    ng2_chars = set(nltk.ngrams(string2, n=3, pad_left=True, pad_right=True, left_pad_symbol=' ', right_pad_symbol=' '))
    if not ng1_chars or not ng2_chars:
        return 0
    return 1 - nltk.jaccard_distance(ng1_chars, ng2_chars)
