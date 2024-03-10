import streamlit as st
import pandas as pd
import joblib
import pickle
import io
import matplotlib.pyplot as plt
import numpy as np
import joblib
import qrcode
import xgboost
from text_utils import contiene_palabra_clave_batch
from text_utils import bootstrap_intervalo_confianza
from limpieza import limpiar_texto

# Descargamos el modelo
modelo = joblib.load('model.sav')
# Usuarios y contraseñas 
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
    uploaded_file = st.file_uploader("Elige un archivo para predecir:", key="file_uploader")
    texto_manual = st.text_area("Introduce el texto que deseé predecir, a continuación presiona Ctrl+Enter:", key="text_area")

    contenido = ""
    # Verifica si se ha cargado un archivo
    if uploaded_file is not None:
        contenido = str(uploaded_file.read(), 'utf-8')
    # Verifica si se ha ingresado texto manualmente
    elif texto_manual != "":
        contenido = texto_manual

    # Muestra el botón de "Predecir" si hay contenido
    if contenido:
        if st.button('Predecir', key='predict_button'):
            texto_limpio = limpiar_texto(contenido)
            prediccion = modelo.predict([texto_limpio])
            frase_prediccion = "Esta gestante SÍ necesita tratamiento antibiotico para prevenir sepsis por EGB." if prediccion[0] == 1 else "Esta gestante NO necesita tratamiento antibiotico para prevenir sepsis por EGB."
            
            st.write('La predicción según el modelo es:')
            st.write(frase_prediccion)
            
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
            
            st.image(buf, caption='Código QR con el resultado de la predicción:')
else:
    usuario = st.sidebar.text_input("Usuario", key="user_input")
    contraseña = st.sidebar.text_input("Contraseña", type="password", key="password_input")
    if st.sidebar.button('Iniciar sesión', key='login_button'):
        if verificar_credenciales(usuario, contraseña):
            st.session_state['autenticado'] = True
            st.rerun()
        else:
            st.sidebar.error("Credenciales incorrectas.")