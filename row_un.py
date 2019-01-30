from utils import *
import collections
from difflib import SequenceMatcher
import time
from fuzzywuzzy import fuzz

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

    # Extract relevant rows with the four choices. Results in much faster execution but lower accuracy
    for choice in choices:
        mx = -1
        can = 0
        for idx, cell in enumerate(table.iloc[:, col]):
            temp = fuzz.ratio(str(cell).strip().lower(), choice)
            if temp>mx:
                mx = temp
                can = idx
        cand_rows.append(can)

    # Find the row with longest common substring
    for idx in cand_rows:
        row = [str(temp) for temp in table.iloc[idx]]
        row = " ".join(row)
        row = " ".join(row.split()).lower()
        l = SequenceMatcher(None, qu, row).find_longest_match(0, len(qu), 0, len(row)).size
        if l > ml:
            ml = l
            r = idx + 1
    if r == int(q[8]):
        count += 1
print(count / len(questions))
time_end = time.time()
print('totally cost', time_end - time_start)
