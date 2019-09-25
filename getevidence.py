import sys, lucene
from os import path, listdir
import csv
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.util import Version
from org.apache.lucene.store import RAMDirectory
from datetime import datetime
from java.io import File 
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser, QueryParserBase
from org.apache.lucene.analysis.core import LowerCaseTokenizer, LowerCaseFilter, StopAnalyzer, StopFilter
from org.apache.lucene.search import BooleanClause

import json
import spacy
nlp = spacy.load('en_vectors_web_lg')
import re

data = {}

#This is how the saved data should look
# {
#   "75397": {
#     "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
#     "label": "SUPPORTS",
#     "evidence": [
#       [
#         "Fox_Broadcasting_Company",
#         0
#       ],
#       [
#         "Nikolaj_Coster-Waldau",
#         7
#       ]
#     ]
#   },
# }

with open('irtest.json') as json_file:  
    data = json.load(json_file)
INPUT_DIR = "wiki-pages-text/"  
INDEX_DIR =  "new-index"  
lucene.initVM() 
index_path = File(INDEX_DIR).toPath() 
directory = SimpleFSDirectory.open(index_path) 

def get_evidence(searcher, analyzer, claim):
    escaped_string = QueryParser.escape(claim)
    query = QueryParser("text", analyzer).parse(escaped_string)
    start = datetime.now()
    scoreDocs = searcher.search(query, 50).scoreDocs
    duration = datetime.now() - start
    claim = nlp(claim)
    claim_evid = []
    line_no = []
    sim_score = []
    final_evidence = []
    for scoreDoc in scoreDocs:
        doc = searcher.doc(scoreDoc.doc)
        norm_doc = doc.get("text")
        norm_doc = nlp(norm_doc)
        val = claim.similarity(norm_doc)
        try:
            int(doc.get("Sno"))
            claim_evid.append(doc.get("keyterm"))
            line_no.append(int(doc.get("Sno")))
            sim_score.append(val)
        except ValueError:
            pass      # or whatever
        
    if len(sim_score)>5:
        for val in range(0,5):
            index = sim_score.index(max(sim_score))
            claim = claim_evid.pop(index)
            line = line_no.pop(index)
            final_evidence.append([claim , line])
            del sim_score[index]
    else:
        for i in range(0, len(sim_score)-1):
            final_evidence.append([claim_evid[i] , int(line_no[i])])
    return final_evidence



searcher = IndexSearcher(DirectoryReader.open(directory))
analyzer = StandardAnalyzer()

for key in data:
    claim = data[key]["claim"]
    data[key]["label"] = " "
    data[key]["evidence"] = get_evidence(searcher, analyzer, claim)


with open('irtest-score.json', 'w') as outfile:  
    json.dump(data, outfile, indent = 4)



