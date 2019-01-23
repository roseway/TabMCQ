from utils import *
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from math import *
import collections


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / denominator, 3)


questions, tables, table_idx = load_data()

row = []
idx2table = {}
tab2idx = {}
i = 0
## title
#for t in table_idx:
#    row.append(table_idx[t])
#    idx2table[i] = t
#    tab2idx[t] = i
#    i += 1

# # keep uppercase header
# for t in table_idx:
#     header = list(tables[t])
#     temp = []
#     for h, c in zip(header, tables[t].iloc[0]):
#         if h.startswith('Unnamed'):
#             temp.append(c)
#         else:
#             tt = ""
#             for let in h:
#                 if let.isupper():
#                     tt += let
#                 else:
#                     break
#             temp.append(tt)
#     row.append(" ".join(temp))
#     idx2table[i] = t
#     i += 1

## replace first row with header
#for t in table_idx:
#    header = list(tables[t])
#    temp = []
#    for h, c in zip(header, tables[t].iloc[0]):
#        if h.startswith('Unnamed'):
#            temp.append(c)
#        else:
#            temp.append(h)
#    row.append(" ".join(temp))
#    idx2table[i] = t
#    tab2idx[t] = i
#    i += 1

# # keep first row
# for t in table_idx:
#     row.append(" ".join(tables[t].iloc[0]))
#     idx2table[i] = t
#     tab2idx[t] = i
#     i += 1

## title + first row
#for t in table_idx:
#    row.append(table_idx[t] + " " + " ".join(tables[t].iloc[0]))
#    idx2table[i] = t
#    tab2idx[t] = i
#    i += 1

## title + first row (replaced)
#for t in table_idx:
#    header = list(tables[t])
#    temp = []
#    for h, c in zip(header, tables[t].iloc[0]):
#        if h.startswith('Unnamed'):
#            temp.append(c)
#        else:
#            temp.append(h)
#    row.append(table_idx[t] + " " + " ".join(temp))
#    idx2table[i] = t
#    tab2idx[t] = i
#    i += 1
    
# title + header + first row
for t in table_idx:
    header = list(tables[t])
    temp = []
    for h, c in zip(header, tables[t].iloc[0]):
        if not h.startswith('Unnamed'):
            temp.append(h)
    row.append(table_idx[t] + " " + " ".join(temp) + " ".join(tables[t].iloc[0]))
    idx2table[i] = t
    tab2idx[t] = i
    i += 1
q_emb = np.loadtxt('q_embedding.txt')

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    row_embeddings = session.run(embed(row))

count1 = 0
count2 = 0
count3 = 0

rec = collections.Counter()
for i in range(len(questions)):
    scores = np.array([cosine_similarity(q_emb[i], t_emb) for t_emb in row_embeddings])
    x = scores.argsort()[::-1]
    if tab2idx[questions[i][7]] in x[:1]:
        count1 += 1
        count2 += 1
        count3 += 1
    elif tab2idx[questions[i][7]] in x[:2]:
        count2 += 1
        count3 += 1
    elif tab2idx[questions[i][7]] in x[:3]:
        count3 += 1
    else:
        rec[questions[i][7]] += 1

print(count1 / len(questions))
print(count2 / len(questions))
print(count3 / len(questions))

print(rec)
