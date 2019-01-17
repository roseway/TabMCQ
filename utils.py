import csv
from os import listdir
import pandas as pd

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
	
