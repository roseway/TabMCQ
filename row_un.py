from utils import *
from difflib import SequenceMatcher
import time

time_start = time.time()

questions, tables, table_idx = load_data()

count = 0

for q in questions:
    table = tables[q[7]]
    que = " ".join(q[0].split()).lower()
    choices = [str(c).lower() for c in q[2:6]]
    pred = -1
    maxlen = -1

    # Find the row with longest common substring
    for idx, row in table.iterrows():
        row = [str(temp) for temp in row]
        row = " ".join(row)
        row = " ".join(row.split()).lower()
        # LCS of row and question
        l1 = SequenceMatcher(None, que, row).find_longest_match(0, len(que), 0, len(row)).size
        # LCS of row and any choice
        l2 = max(SequenceMatcher(None, choice, row).find_longest_match(0, len(choice), 0, len(row)).size for choice in choices)
        if l1 + l2 > maxlen:
            maxlen = l1 + l2
            pred = idx + 1

    if pred == int(q[8]):
        count += 1

print('Accuracy is', count / len(questions))
time_end = time.time()
print('Time cost', time_end - time_start)
