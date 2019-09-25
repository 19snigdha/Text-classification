import spacy
import numpy as np
nlp = spacy.load('en_vectors_web_lg')
import json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import to_categorical
import pandas as pd
from keras import backend as K
from keras import layers, Model, models
from keras.models import model_from_json
from keras.models import load_model


train_data = pd.read_csv('irtest-score.csv')# csv for prediction
claim_data = (train_data["Claim"].fillna(' ')).tolist()
evidence_data = (train_data["Evidence"].fillna(' ')).tolist()
with open('irtest-score.json') as json_f:  
    final_predict = json.load(json_f)

json_file = open('old-model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("old-model1.h5")
print("Loaded model from disk")
 

model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

def generate_data_vectors(nlp, claim, evidence, random, max_length):
    sents = claim + evidence
    sentence= []
    for sent in sents:
        doc = nlp(sent)
        words = []
        for i, token in enumerate(doc):
            if token.has_vector and token.vector_norm == 0: continue
            if i > max_length:break
            if token.has_vector: words.append(token.rank + random + 1)
            else: words.append(token.rank % random + 1)
        vector_word = np.zeros((max_length), dtype="int")

        crop = min(max_length, len(words))
        vector_word[:crop] = words[:crop]
        sentence.append(vector_word)
    return [np.array(sentence[: len(claim)]), np.array(sentence[len(claim) :])]

data = generate_data_vectors(nlp, claim_data, evidence_data, 100, 1000)



Xnew = [data[0], data[1]]
print(len(Xnew))
# make a prediction
ynew = model.predict([data[0], data[1]])
dic_list = ['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES']
print(len(ynew), 'this len')

i = 0
for key in final_predict:
	print(i)
	y_classes = np.argmax(ynew[i])
	final_predict[key]["label"] = dic_list[y_classes]
	i += 1

with open('irtest-score.json', 'w') as outfile:  
    json.dump(final_predict, outfile, indent=4)





