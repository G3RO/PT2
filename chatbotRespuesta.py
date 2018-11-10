# NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# TF
import numpy as np
import tflearn
import tensorflow as tf
import random

import pickle

data = pickle.load( open( "training_data", "rb" ) )

palabras = data['palabras']
clases = data['clases']
#train_x = data['train_x']
#train_y = data['train_y']

tflearn.models.dnn.DNN.load(open('./model.tflearn.index'))

import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

def clean_up_sentence(oracion):
    oracion_palabras = nltk.word_tokenize(oracion)
    oracion_palabras = [stemmer.stem(word.lower()) for word in oracion_palabras]
    return oracion_palabras

#def bow(oracion, palabras, show_details=False):
    #oracion_palabras = clean_up_sentence(oracion)
    ##oracion_palabras = nltk.word_tokenize(oracion)
    #bag = [0]*len(palabras)  
    #for s in oracion_palabras:
        #for i,w in enumerate(palabras):
            #if w == s: 
                #bag[i] = 1
                #if show_details:
                    #print ("Encontrada en el bag: %s" % w)

    #return(np.array(bag))

#def response(oracion, userID='123', show_details=true):
    #resultados = classify(oracion)
    # Si se tiene la clasificación, buscala en los intents
    #if resultados:
        #while resultados:
            #for i in intents['intents']:
                #if i['tag'] == resultados[0][0]:
                    # Respuesta aleatoria
                    #return print(random.choice(i['responses']))

            #resultados.pop(0)

context = {}

ERROR_THRESHOLD = 0.25
def classify(oracion):
    resultados = model.predict([bow(oracion, palabras)])[0]
    resultados = [[i,r] for i,r in enumerate(resultados) if r>ERROR_THRESHOLD]
    resultados.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in resultados:
        return_list.append((clases[r[0]], r[1]))
    return return_list

def response(oracion, userID='123', show_details=False):
    resultados = classify(oracion)
    if resultados:
        while resultados:
            for i in intents['intents']:
                if i['tag'] == resultados[0][0]:
                    # contexto al intent
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        return print(random.choice(i['responses']))

            resultados.pop(0)
            
classify('La tienda esta abierta?')
response('La tienda esta abierta?')
response('Busco unos tenis rojos')
response('Buenas tardes!')
response('Hasta luego')