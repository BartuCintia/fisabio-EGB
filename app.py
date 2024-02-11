import streamlit as st
import pandas as pd
import joblib
import pickle
import io
import matplotlib.pyplot as plt
import numpy as np
import joblib
import qrcode
from text_utils import contiene_palabra_clave_batch
from text_utils import bootstrap_intervalo_confianza
from limpieza import limpiar_texto

# Des
modelo = joblib.load('model.sav')
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
            frase_prediccion = ""
            if prediccion[0] == 1:
                frase_prediccion = "Según el modelo SVC, SÍ posee la enfermedad de Streptococcus agalactiae."
            else:
                frase_prediccion = "Según el modelo SVC, NO posee la enfermedad de Streptococcus agalactiae."
            
            st.write('La predicción según el modelo es:')
            st.write(frase_prediccion)
            
            # Generación del código QR con el texto de la predicción
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(frase_prediccion)
            qr.make(fit=True)
            img_qr = qr.make_image(fill_color="black", back_color="white")
            buf = io.BytesIO()
            img_qr.save(buf)
            buf.seek(0)
            
            st.image(buf, caption='Código QR con el resultado de la predicción')


else:
    usuario = st.sidebar.text_input("Usuario")
    contraseña = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button('Iniciar sesión'):
        if verificar_credenciales(usuario, contraseña):
            st.session_state['autenticado'] = True
            st.rerun()
        else:
            st.sidebar.error("Credenciales incorrectas.")




