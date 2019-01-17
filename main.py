from utils import *

questions, tables, table_idx = load_data()
print(questions[0])
#n = 0
#for q in questions:
#	res = -1
#	mxm = -1
#	choices = [str(c).lower() for c in q[2:6]]
#	table = tables[q[7]]
#	for idx, col in enumerate(table):
#		count = 0
#		for cell in table[col]:
#			temp = str(cell).strip().lower()
#			for c in choices:
#				if temp == c:
#					count += 1
#		if count > mxm:
#			mxm = count
#			res = idx
#	if res == int(q[-1]):
#		n+=1
#	else:
#		print(q, res)


#print(n/len(questions))

