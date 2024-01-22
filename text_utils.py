from sklearn.utils import resample
import numpy as np
from sklearn.metrics import accuracy_score

# Lista de términos para describir la enfermedad
palabras_clave = [ "streptococcus", "estreptoco"]

# Verificamos la presencia de palabras clave en un conjunto de textos
def contiene_palabra_clave_batch(textos):
    return [[contiene_palabra_clave(texto)] for texto in textos]

# Verificamos la presencia de palabras clave en un solo texto
def contiene_palabra_clave(texto):
    return any(palabra in texto for palabra in palabras_clave)

#
def bootstrap_intervalo_confianza(modelo, X, n=1000, alpha=0.05):
    predicciones = np.zeros((n, len(X)))
    
    for i in range(n):
        X_resample = resample(X)
        y_pred = modelo.predict(X_resample)
        predicciones[i] = y_pred

    # Calcula el intervalo de confianza
    lower = np.percentile(predicciones, 100 * alpha / 2.)
    upper = np.percentile(predicciones, 100 * (1 - alpha / 2.))

    return lower, upper
