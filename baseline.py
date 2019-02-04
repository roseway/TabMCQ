from utils import *
import tensorflow as tf
import tensorflow_hub as hub
from math import *
from difflib import SequenceMatcher
import time
import collections
import numpy as np


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / denominator, 3)


time_start = time.time()

questions, tables, table_idx = load_data()
test = ['regents-01', 'regents-05&09', 'regents-06', 'regents-13', 'regents-22', 'regents-25&26', 'regents-40',
        'regents-41', 'monarch-45', 'monarch-46', 'monarch-47', 'monarch-49', 'monarch-54', 'monarch-56']

# Table selection
row = []
idx2table = {}
tab2idx = {}
i = 0

# title + header + first row
for t in table_idx:
    if t in test:
        header = list(tables[t])
        temp = []
        for h, c in zip(header, tables[t].iloc[0]):
            if not h.startswith('Unnamed'):
                temp.append(h)
        row.append(table_idx[t] + " " + " ".join(temp) + " " + " ".join(tables[t].iloc[0]))
        idx2table[i] = t
        tab2idx[t] = i
        i += 1

q_emb = np.loadtxt('data/q_embedding.txt')

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

# Compute table representation
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    row_embeddings = session.run(embed(row))

count = 0
correct = 0
wrong = collections.Counter()
for i in range(len(questions)):
    q = questions[i]
    if q[7] in test:
        count += 1
        scores = np.array([cosine_similarity(q_emb[i], t_emb) for t_emb in row_embeddings])
        x = scores.argsort()[::-1]
        que = " ".join(q[0].split()).lower()
        choices = [str(c).lower() for c in q[2:6]]
        ans = -1
        maxlen = -1
        for idx in x[:3]:
            table = tables[idx2table[idx]]
            # Find the row with longest common substring
            for idx, row in table.iterrows():
                row = [str(temp) for temp in row]
                row = " ".join(row)
                row = " ".join(row.split()).lower()
                # LCS of row and question
                l1 = SequenceMatcher(None, que, row).find_longest_match(0, len(que), 0, len(row)).size
                # LCS of row and any choice
                cc = [SequenceMatcher(None, choice, row).find_longest_match(0, len(choice), 0, len(row)).size for choice
                      in
                      choices]
                if l1 + max(cc) > maxlen:
                    maxlen = l1 + max(cc)
                    ans = np.argmax(np.array(cc)) + 1

        if ans == int(q[6]):
            correct += 1
        else:
            wrong[q[7]] += 1
print('Accuracy is', correct / count)
print(wrong)

# # Just table selection
# count = 0
# correct = 0
# wrong = collections.Counter()
# rec = collections.Counter()
# count1 = 0
# count2 = 0
# count3 = 0
# for i in range(len(questions)):
#     if questions[i][7] in test:
#         rec[questions[i][7]] += 1
#         count += 1
#         scores = np.array([cosine_similarity(q_emb[i], t_emb) for t_emb in row_embeddings])
#         x = scores.argsort()[::-1]
#         if tab2idx[questions[i][7]] in x[:1]:
#             count1 += 1
#             count2 += 1
#             count3 += 1
#         elif tab2idx[questions[i][7]] in x[:2]:
#             count2 += 1
#             count3 += 1
#         elif tab2idx[questions[i][7]] in x[:3]:
#             count3 += 1
#         else:
#             wrong[questions[i][7]] += 1
# print(count1 / count)
# print(count2 / count)
# print(count3 / count)
# print(wrong)
# for t in test:
#     print(t, wrong[t] / rec[t])

# # Just answer selection
# for i in range(len(questions)):
#     count += 1
#     q = questions[i]
#     rec[q[7]] += 1
#     que = " ".join(q[0].split()).lower()
#     choices = [str(c).lower() for c in q[2:6]]
#     ans = -1
#     maxlen = -1
#     relerow = -1
#     table = tables[q[7]]
#     # Find the row with longest common substring
#     for idx, row in table.iterrows():
#         row = [str(temp) for temp in row]
#         row = " ".join(row)
#         row = " ".join(row.split()).lower()
#         # LCS of row and question
#         l1 = SequenceMatcher(None, que, row).find_longest_match(0, len(que), 0, len(row)).size
#         # LCS of row and any choice
#         cc = [SequenceMatcher(None, choice, row).find_longest_match(0, len(choice), 0, len(row)).size for choice in
#               choices]
#         if l1 + max(cc) > maxlen:
#             maxlen = l1 + max(cc)
#             ans = np.argmax(np.array(cc)) + 1
#             relerow = idx
#
#     if ans == int(q[6]):
#         correct += 1
#     else:
#         #print(q[0], q[ans + 1], " ".join(table.loc[int(q[8]) - 1]), " ".join(table.loc[relerow]))
#         wrong[q[7]] += 1
# print(wrong)
# print(correct/count)
# for t in table_idx:
#     print(t, wrong[t]/rec[t])
#
#
# print('Accuracy is', count / len(questions))
# print(wrong)

time_end = time.time()
print('Time cost', time_end - time_start)
