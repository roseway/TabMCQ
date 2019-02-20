__author__ = 'Nick Hirakawa'

from bm25.invdx import build_data_structures
from bm25.rank import score_BM25
import collections
import operator
from math import log

class QueryProcessor:
	def __init__(self, queries, corpus):
		self.queries = queries
		self.index, self.dlt = build_data_structures(corpus)

	def run(self):
		results = []
		for query in self.queries:
			results.append(self.run_query(query))
		return results

	def run_query(self, query):
		query_result = collections.Counter()
		for term in query:
			if term in self.index:
				doc_dict = self.index[term] # retrieve index entry
				for docid, freq in doc_dict.items(): #for each document and its word frequency
					score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt),
									   dl=self.dlt.get_length(docid), avdl=self.dlt.get_average_length()) # calculate score
					query_result[docid] += score

		return query_result
		
	def idf(self):
		idfs = []
		for query in self.queries:
			score = 0
			for term in query:
				if term in self.index:
					doc_dict = self.index[term] # retrieve index entry
					score += log( (len(self.dlt) - len(doc_dict) + 0.5) / (len(doc_dict) + 0.5)) # calculate score
			idfs.append(score)
		return idfs
