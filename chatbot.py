# NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# TF
import numpy as np
import tflearn
import tensorflow as tf
import random

# Estructura (pickle)
import pickle

# Intents: operaciones que queramos accionar
# Json file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Contenedores    
palabras = []
clases = []
documentos = []
ign_palabras = ['?']

# Loopear sobre cada oración en los intents (patrones)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Hacer cada palabra en la oración un token
        w = nltk.word_tokenize(pattern)
        # Agrega cada token a la lista
        palabras.extend(w)
        # Agrega los documentos al corpus
        documentos.append((w, intent['tag']))
        # Agrega a la lista clases
        if intent['tag'] not in clases:
            clases.append(intent['tag'])

# hacer stem y quitar duplicados
palabras = [stemmer.stem(w.lower()) for w in palabras if w not in ign_palabras]
palabras = sorted(list(set(palabras)))

# quitar duplicados
clases = sorted(list(set(clases)))

print (len(documentos), "documentos")
print (len(clases), "clases", clases)
print (len(palabras), "palabras stemmed unicas", palabras)

# creamos nuestro training set
training = []
salida = []

# array vacio para la lista de salidas
salida_vacia = [0] * len(clases)

# Training set, bag de palabras para cada oración
for doc in documentos:
    bag = []
    # Lista que contiene los patrones de las palabras ya tokenizadas
    patrones_palabras = doc[0]
    # stem a las palabras
    patrones_palabras = [stemmer.stem(word.lower()) for word in patrones_palabras]
    # array contenedor de palabras
    for w in palabras:
        bag.append(1) if w in patrones_palabras else bag.append(0)

    # las salidas son 0 en cada tag y 1 en el actual tag
    output_row = list(salida_vacia)
    output_row[clases.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

# train y test
train_x = list(training[:,0])
train_y = list(training[:,1])

# reset
tf.reset_default_graph()

# 2 capas con softmax
red_neuronal = tflearn.input_data(shape=[None, len(train_x[0])])
red_neuronal = tflearn.fully_connected(red_neuronal, 8)
red_neuronal = tflearn.fully_connected(red_neuronal, 8)
red_neuronal = tflearn.fully_connected(red_neuronal, len(train_y[0]), activation='softmax')
red_neuronal = tflearn.regression(red_neuronal)

# Modelo: DNN
model = tflearn.DNN(red_neuronal, tensorboard_dir='tflearn_logs')

model.save('model.tflearn')

# Entrenamiento (gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)

def clean_up_sentence(oracion):
    oracion_palabras = nltk.word_tokenize(oracion)
    oracion_palabras = [stemmer.stem(word.lower()) for word in oracion_palabras]
    return oracion_palabras

# Regresa la bag de palabras (array), 0 o 1 para cada palabra en la bag que exista en la oración
def bow(oracion, palabras, show_details=False):
    oracion_palabras = clean_up_sentence(oracion)
    bag = [0]*len(palabras)  
    for s in oracion_palabras:
        for i,w in enumerate(palabras):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("Encontrada en el bag: %s" % w)

    return(np.array(bag))

p = bow("Hola, busco unos tenis de color rojo", palabras)
print (p)
print (clases)
print (model.predict([p]))

# guarda en estructura
pickle.dump( {'palabras':palabras, 'clases':clases, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )