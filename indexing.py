import sys 
import lucene 
from os import path, listdir 
from java.io import File 
from org.apache.lucene.document import Document, Field, StringField, TextField 
from org.apache.lucene.util import Version 
from org.apache.lucene.store import SimpleFSDirectory 
from datetime import datetime 
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer 
from org.apache.lucene.analysis.standard import StandardAnalyzer 
from org.apache.lucene.index import IndexWriter, IndexWriterConfig 

           
           
INPUT_DIR = "wiki-pages-text/"  
INDEX_DIR =  "new-index"   

           
def create_document(line): 
    doc = Document()
    line = line.split() 
    keyterm = line[0] 
    doc.add(StringField( "keyterm", keyterm, Field.Store.YES)) 
    index = line[1] 
    doc.add(StringField( "Sno" , index, Field.Store.YES)) 
    del line[0:2]
    line = ' '.join(line) 
    qterm = keyterm.replace("_"," ") 
    if qterm not in line: 
        line = qterm + ' ' + line 
    doc.add(TextField( "text" ,line, Field.Store.YES)) 
    return doc 
               

lucene.initVM() 
index_path = File(INDEX_DIR).toPath() 
directory = SimpleFSDirectory.open(index_path) 
analyzer = StandardAnalyzer() 
config = IndexWriterConfig(analyzer) 
writer = IndexWriter(directory, config) 
print("Number of documents:", writer.numDocs()) 

for input_file in listdir(INPUT_DIR):
    print("Current file:"  , input_file) 
    if input_file.endswith(".txt"):
        path = INPUT_DIR+input_file 
        with open(path) as file:
            line = file.readline()
            while(line) :
                line = file.readline()
                if len(line.strip()) != 0 :
                    doc = create_document(line) 
                    writer.addDocument(doc)  
        file.close() 
print("finally:",writer.numDocs()) 
print("Indexing done!") 
