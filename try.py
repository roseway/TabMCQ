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
from sklearn.utils import shuffle
from fuzzywuzzy import fuzz

questions, tables, table_idx = load_data()

# Match cells with answer
count = 0
for idx, q in enumerate(questions):
	qu = q[0].lower()
	cell = str(tables[q[7]].iat[int(q[8])-1, int(q[9])]).strip().lower()
	choices = [str(c).lower() for c in q[2:6]]
	res = 0
	ratio = -float('inf')
	for i in range(4):
		temp = fuzz.ratio(cell, choices[i])
		if  temp > ratio:
			ratio = temp
			res = i + 1
	if res == int(q[6]):
		count += 1
	else:
		print(idx, cell, choices)
print(count/len(questions))		# 96%. Most wrong cases are because of wrong annotations

#Check if value in question
#counter = collections.Counter()
#count = 0
#for q in questions:
#	qu = q[0].lower()
#	table = tables[q[7]]
#	row = table.loc[int(q[8])-1]
#	flag = 0
#	for h in list(table):
#		if not h.startswith('Unnamed'):
#			if str(row[h]).lower().strip()[:-1] in qu:
#				count += 1
#				flag = 1
#				break
#	if not flag:
#		counter[q[7]] += 1
#		print(qu, q[7])

#print(counter)
#print(count/len(questions))
