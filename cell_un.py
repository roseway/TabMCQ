from utils import *
from fuzzywuzzy import fuzz

questions, tables, table_idx = load_data()

# Match cells with answer
count = 0
for idx, q in enumerate(questions):
	cell = str(tables[q[7]].iat[int(q[8])-1, int(q[9])]).strip().lower()
	choices = [str(c).lower() for c in q[2:6]]
	res = 0
	ratio = -float('inf')
	for i in range(4):
		temp = fuzz.ratio(cell, choices[i])
		if temp > ratio:
			ratio = temp
			res = i + 1
	if res == int(q[6]):
		count += 1
	else:
		print(idx, cell, choices)
print(count/len(questions))		# 96%. Most wrong cases are because of wrong annotations

