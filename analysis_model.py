from limpieza import limpiar_texto
import numpy as np
import pickle

from text_utils import contiene_palabra_clave_batch
from sklearn.feature_extraction.text import TfidfVectorizer

def calcular_log_probabilidades(probabilidades):
    # Utilizamos logaritmo natural para calcular log probabilidades
    return np.log(probabilidades)

def calcular_ratio_probabilidad(probabilidades):
    # Asumiendo un modelo binario, calculamos el ratio de la clase 1 sobre la clase 0
    # Evitamos división por cero agregando un pequeño valor epsilon
    epsilon = 1e-9
    return probabilidades[:, 1] / (probabilidades[:, 0] + epsilon)

def calcular_puntuacion_confianza(probabilidades):
    # Diferencia entre las probabilidades de las dos clases
    return np.abs(probabilidades[:, 1] - probabilidades[:, 0])

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Paso 1: Cargar el modelo
with open('model_nb.pkl', 'rb') as file:
    modelo_nb = pickle.load(file)
tfidf_vectorizer = modelo_nb.named_steps['features'].transformer_list[0][1]
vocabulario = tfidf_vectorizer.get_feature_names_out()
log_probabilidades = modelo_nb.named_steps['clf'].feature_log_prob_
num_vocabulario = len(vocabulario)

# Paso 3: Analizar características influyentes
for i, clase in enumerate(modelo_nb.classes_):
    print(f"Clase {clase}:")
    log_prob_clase = log_probabilidades[i]
    
    # Asegurar que los índices no excedan el número de términos en el vocabulario
    indices_validos = [indice for indice in log_prob_clase.argsort() if indice < num_vocabulario]
    
    terminos_influyentes = [(vocabulario[indice], log_prob_clase[indice]) for indice in indices_validos[-10:]]  # Top 10 términos
    print(terminos_influyentes)



