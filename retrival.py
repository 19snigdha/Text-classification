
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

data = {}
with open('irtest-score.json') as json_file:  
    data = json.load(json_file)

claims = []
evidence = []
label = []
Sno = []

INPUT_DIR = "wiki-pages-text/"  
INDEX_DIR =  "new-index"  
lucene.initVM() 
index_path = File(INDEX_DIR).toPath() 
directory = SimpleFSDirectory.open(index_path) 


def getTrainingData(searcher, analyzer, Sno, keyterm):
    query = str(Sno) + ' ' + keyterm
    escaped_string = MultiFieldQueryParser.escape(query)
    multiQueryParser = MultiFieldQueryParser.parse(escaped_string, ["Sno", "keyterm"], [BooleanClause.Occur.SHOULD, BooleanClause.Occur.SHOULD],analyzer)
    start = datetime.now()
    scoreDocs = searcher.search(multiQueryParser, 1).scoreDocs
    duration = datetime.now() - start
    result =''
    for scoreDoc in scoreDocs:
        doc = searcher.doc(scoreDoc.doc)
        result = str(doc.get("text"))
    return result
        

searcher = IndexSearcher(DirectoryReader.open(directory))
analyzer = WhitespaceAnalyzer()
result = {}

for key in data:
    evid = data[key]["evidence"]
    claims.append(data[key]["claim"])
    label.append(data[key]["label"])
    result = ''
    if len(evid) == 0:
        evidence.append(result)
    else:
        for val in evid:
            if len(val)>0:
                qval = getTrainingData(searcher, analyzer, str(val[1]), val[0])
                result = result + qval + ' '
        evidence.append(result)

for c, e, l in zip(claims, label, evidence):
    fileval = [c,e,l]
    with open('irtest-score.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(fileval)

csvFile.close()   


