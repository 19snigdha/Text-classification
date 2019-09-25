import spacy
import numpy as np
nlp = spacy.load('en_vectors_web_lg')
import json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import to_categorical
from keras import backend as K
from keras import layers, Model, models

import pandas as pd
MAX_DATA = 12000

def get_ir_data():
    train_data = pd.read_csv('finalTrain1.csv') # training data in csv format
    claim_data = (train_data["Claim"].fillna(' ')).tolist()
    evidence_data = (train_data["Evidence"].fillna(' ')).tolist()
    claim_data= claim_data[:MAX_DATA]
    evidence_data = evidence_data[:MAX_DATA]
    return claim_data, evidence_data, train_data

def map_labels(train_data):
    dic = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 1, 'REFUTES': 2}
    labels = (train_data["Label"]).tolist()
    labels = [dic[v] for v in labels]
    labels = labels[:MAX_DATA]
    labels = to_categorical(np.asarray(labels, dtype='int32'))
    return labels

def generate_data_vectors (nlp, claim, evidence, random, max_length):
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

claim_data, evidence_data, train_data = get_ir_data()
labels = map_labels(train_data)
data = generate_data_vectors(nlp, claim_data, evidence_data, 100, 1000)

def get_embeddings(vocab, embedd_dim):
    length = max(lex.rank for lex in vocab)+2
    out_fo_vocab = np.random.normal(size=(embedd_dim, vocab.vectors_length))
    out_fo_vocab = out_fo_vocab / out_fo_vocab.sum(axis=1, keepdims=True)   
    embedding_matrix = np.zeros((length + embedd_dim, vocab.vectors_length), dtype='float32')
    embedding_matrix[1:(embedd_dim + 1), ] = out_fo_vocab
    for word in vocab:
        if word.has_vector and word.vector_norm > 0:
            embedding_matrix[embedd_dim + word.rank + 1] = word.vector / word.vector_norm 
    return embedding_matrix


embeddings = get_embeddings(nlp.vocab, 100)

def generate_embedding(vectors, max_length, projected_dim):
    return models.Sequential([layers.Embedding(vectors.shape[0], vectors.shape[1], input_length=max_length, weights=[embeddings], trainable=False),
        layers.TimeDistributed(layers.Dense(projected_dim, activation=None, use_bias=False))
    ])

def feedforward(num_units=200):
    return models.Sequential([layers.Dense(num_units, activation='relu'),layers.Dropout(0.2),layers.Dense(num_units, activation='relu'),layers.Dropout(0.2)
    ])

def vect_normalizer(axis):
    def _normalize(att_weights):
        exponent = K.exp(att_weights)
        sum_weights = K.sum(exponent, axis=axis, keepdims=True)
        return exponent/sum_weights
    return _normalize

def get_sum(x):
    return K.sum(x, axis=1)

def aggregate(att_weights, sent_claim, sent_evid, feed_forward_2):
    norm_sent_claim_weights = layers.Lambda(vect_normalizer(1))(att_weights)
    norm_sent_evid_weights = layers.Lambda(vect_normalizer(2))(att_weights)
    alpha = layers.dot([norm_sent_claim_weights, sent_claim], axes=1)
    beta  = layers.dot([norm_sent_evid_weights, sent_evid], axes=1)
    comp1 = layers.concatenate([sent_claim, beta])
    comp2 = layers.concatenate([sent_evid, alpha])
    vect1 = layers.TimeDistributed(feed_forward_2)(comp1)
    vect2 = layers.TimeDistributed(feed_forward_2)(comp2)
    vect1_sum = layers.Lambda(get_sum)(vect1)
    vect2_sum = layers.Lambda(get_sum)(vect2)
    concat = layers.concatenate([vect1_sum, vect2_sum])
    return concat


def decompose_attention_model(vectors, num_hidden, num_classes, projected_dim):
    
    sentence_claim = layers.Input(shape=(1000,), dtype='int32', name='sentence-claim')
    sentence_evidence = layers.Input(shape=(1000,), dtype='int32', name='sentence-evidence')
    word_embed = generate_embedding(vectors, 1000, projected_dim)
    sent_claim = word_embed(sentence_claim)
    sent_evid = word_embed(sentence_evidence)
    feed_forward_1 = feedforward(num_hidden)
    att_weights = layers.dot([feed_forward_1(sent_claim), feed_forward_1(sent_evid)], axes=-1)
    feed_forward_2 = feedforward(num_hidden)
    concat = aggregate(att_weights, sent_claim, sent_evid, feed_forward_2)
    feed_forward_3 = feedforward(num_hidden)
    out = feed_forward_3(concat)
    out = layers.Dense(num_classes, activation='softmax')(out)
    model = Model([sentence_claim, sentence_evidence], out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


K.clear_session()
m = decompose_attention_model(embeddings, 200, 3, 200)
m.summary()
m.fit([data[0], data[1]], labels, batch_size=256, epochs=1, validation_split=.2)
scores = m.evaluate([data[0], data[1]], labels, verbose=0)
print("%s: %.2f%%" % (m.metrics_names[1], scores[1]*100))
 
model_json = m.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
m.save_weights("model.h5")
print("Saved model to disk")

