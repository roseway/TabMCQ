from utils import *
from difflib import SequenceMatcher
import nltk
import string
import collections
import random

questions, tables, table_idx = load_data()
random.seed(0)
# np.random.seed(0)
random.shuffle(questions)

punc_table = str.maketrans({key: None for key in string.punctuation})

tablepre = []
with open("tablepre.txt") as f:
    for line in f.readlines():
        x = line[2:-2].split('\'')
        tablepre.append([x[0], x[2]])

correct = 0
iii = 0
count = 0
wrong = collections.Counter()
allrec = collections.Counter()
for q in questions[8200:]:
    count += 1
    q_tok = nltk.word_tokenize(q[0].translate(punc_table).lower())
    a_tok = [nltk.word_tokenize(a.translate(punc_table).lower()) for a in q[2:6]]
    table = tables[tablepre[iii][0]]
    cells = table.applymap(str).values
    max_tok = -1
    # Find the row with most common tokens
    for row in cells:
        match_tok = 0
        line = set(nltk.word_tokenize(" ".join(row).translate(punc_table).lower()))
        for tok in q_tok:
            if tok in line:
                match_tok += 1
        for i in range(4):
            for tok in a_tok[i]:
                if tok in line:
                    match_tok += 1
            if max_tok < match_tok:
                ans = i + 1
                max_tok = match_tok
    if ans == int(q[6]):
        correct += 1
    else:
        wrong[q[7]] += 1
        # print(iii, q[0], q[ans+1], q[int(q[6])+1])
    iii += 1
print(correct/count)

correct = 0
count = 0
wrong = collections.Counter()
rec = collections.Counter()
for q in questions[8200:]:
    count += 1
    rec[q[7]] += 1
    que = " ".join(q[0].split()).lower()
    choices = [str(c).lower() for c in q[2:6]]
    ans = -1
    maxlen = -1
    relerow = -1
    table = tables[tablepre[count-1][0]]
    # Find the row with longest common substring
    for idx, row in table.iterrows():
        row = [str(temp) for temp in row]
        row = " ".join(row)
        row = " ".join(row.split()).lower()
        # LCS of row and question
        l1 = SequenceMatcher(None, que, row).find_longest_match(0, len(que), 0, len(row)).size
        # LCS of row and any choice
        cc = [SequenceMatcher(None, choice, row).find_longest_match(0, len(choice), 0, len(row)).size for choice in
              choices]
        if l1 + max(cc) > maxlen:
            maxlen = l1 + max(cc)
            ans = np.argmax(np.array(cc)) + 1
            relerow = idx

    if ans == int(q[6]):
        correct += 1
    else:
        #print(q[0], q[ans + 1], " ".join(table.loc[int(q[8]) - 1]), " ".join(table.loc[relerow]))
        wrong[q[7]] += 1
print(correct/count)
