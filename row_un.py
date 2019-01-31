from utils import *
import collections
from difflib import SequenceMatcher
import time

time_start = time.time()

questions, tables, table_idx = load_data()

acc = []
wrong = collections.Counter()
count = 0
for q in questions:
    r = -1
    ml = -1
    table = tables[q[7]]
    qu = " ".join(q[0].split()).lower()
    choices = [str(c).lower() for c in q[2:6]]
    col = int(q[9])
    cand_rows = []

    # Find the row with longest common substring
    for idx, row in table.iterrows():
        row = [str(temp) for temp in row]
        row = " ".join(row)
        row = " ".join(row.split()).lower()
        l = SequenceMatcher(None, qu, row).find_longest_match(0, len(qu), 0, len(row)).size
        tt = 0
        for choice in choices:
            tt = max(tt, SequenceMatcher(None, choice, row).find_longest_match(0, len(choice), 0, len(row)).size)
        if l + tt > ml:
            ml = l + tt
            r = idx + 1
    if r == int(q[8]):
        count += 1
print(count / len(questions))
time_end = time.time()
print('Time cost', time_end - time_start)
