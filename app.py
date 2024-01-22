import streamlit as st
import pandas as pd
import joblib
import pickle
from wordcloud import WordCloud
import io
import matplotlib.pyplot as plt
import numpy as np
from text_utils import contiene_palabra_clave_batch
from text_utils import bootstrap_intervalo_confianza
from limpieza import limpiar_texto

modelo = pickle.load(open('model_nb.pkl', 'rb'))
vectorizer = modelo.named_steps['features'].transformer_list[0][1]

def analizar_wordcloud(modelo, texto_procesado, vectorizer, top_n):
    # Extraer el modelo Naive Bayes del pipeline
    naive_bayes_modelo = modelo.named_steps['clf']

    # Obtener las clases y el recuento de características
    clases = naive_bayes_modelo.classes_
    feature_count = naive_bayes_modelo.feature_count_
    
    # Transformar el texto procesado utilizando el vectorizador
    texto_vectorizado = vectorizer.transform([texto_procesado])

    contribuciones = {}
    for idx, clase in enumerate(clases):
        # Calcular la importancia relativa de cada característica para esta clase
        log_probabilidades = np.log(feature_count[idx, :] + 1)  # Se agrega 1 para evitar log(0)
        # Usar get_feature_names_out() en lugar de get_feature_names()
        contribuciones_clase = {vectorizer.get_feature_names_out()[i]: log_probabilidades[i] for i in texto_vectorizado.nonzero()[1]}
        contribuciones[clase] = contribuciones_clase
    contribuciones_combinadas = {}
    for contribuciones_clase in contribuciones.values():
        for palabra, importancia in contribuciones_clase.items():
            contribuciones_combinadas[palabra] = contribuciones_combinadas.get(palabra, 0) + importancia

    # Seleccionar las top_n palabras con mayor importancia
    palabras_top = sorted(contribuciones_combinadas.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return dict(palabras_top)

usuarios = {
    "admin": "admin12",
    "usuario1": "contraseña1",
    "usuario2": "contraseña2"
}

def verificar_credenciales(usuario, contraseña):
    return usuario in usuarios and usuarios[usuario] == contraseña

# Inicio de sesión
if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False

if st.session_state['autenticado']:
    st.title('Predicción con Modelo de Machine Learning')
    uploaded_file = st.file_uploader("Elige un archivo para predecir")

    if uploaded_file is not None:
        # Leer el contenido del archivo
        contenido = str(uploaded_file.read(), 'utf-8')

        # Limpiar el texto
        texto_limpio = limpiar_texto(contenido)

        # Predicción
        if st.button('Predecir'):
            prediccion = modelo.predict([texto_limpio])
            y_score = modelo.predict_proba([texto_limpio])
            datos_wordcloud = analizar_wordcloud(modelo, texto_limpio, vectorizer, top_n=15)

            # Crear el wordcloud
            wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate_from_frequencies(datos_wordcloud)

            # Configurar el gráfico de matplotlib
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")

            # Guardar la figura en un buffer de bytes
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            # Mostrar el wordcloud en Streamlit
            st.write(f'La predicción según el modelo es:')
            if prediccion[0] == 1:
                st.write(f'Según el modelo de Naive Bayes, Sí poseé la enfermedad de Streptococcus agalactiae.')
            else:
                st.write(f'Según el modelo de Naive Bayes, NO poseé la enfermedad de Streptococcus agalactiae.')
            st.write(f'La probabilidad obtenida en las dos clases es:')
            st.write(f'La probabilidad de que tenga la enfermedad es {y_score[0][0]} y la probabilidad de que SÍ {y_score[0][1]}\n')
            st.write(f'Imagen del conjunto de palabras que se tuvieron más en cuenta a la hora de la decisión de predecir el texto:')
            st.image(buf, use_column_width=True)

else:
    usuario = st.sidebar.text_input("Usuario")
    contraseña = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button('Iniciar sesión'):
        if verificar_credenciales(usuario, contraseña):
            st.session_state['autenticado'] = True
            st.rerun()
        else:
            st.sidebar.error("Credenciales incorrectas.")




