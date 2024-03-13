import streamlit as st
import os
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
    '''
        Función para verificar las credenciales de un usuario.

        Argumentos:
            usuario (str): El nombre de usuario proporcionado.
            contraseña (str): La contraseña proporcionada.

        Devuelve:
            bool: True si las credenciales son válidas, False de lo contrario.
    '''
    return usuario in usuarios and usuarios[usuario] == contraseña


# Inicio de sesión
if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False

if st.session_state['autenticado']:
    st.title('Sistema eSalud:')
    st.header('Identificación, almacenamiento y recuperación de información crítica en la atención a la gestante en el área de urgencias de maternidad.')

    uploaded_file = st.file_uploader("Elige un archivo para predecir:", key="file_uploader")
    gestante_nombre = st.text_input("Nombre o identificador de la gestante:", key="gestante_nombre_input")
    texto_manual = st.text_area("Introduce el texto que deseé predecir, a continuación presiona Ctrl+Enter:", key="text_area")

    contenido = ""
    # Verifica si se ha cargado un archivo
    if uploaded_file is not None:
        contenido = str(uploaded_file.read(), 'utf-8')
        # Asigna el nombre del archivo como nombre de la gestante
        gestante_nombre = os.path.splitext(uploaded_file.name)[0]
    # Verifica si se ha ingresado texto manualmente
    elif texto_manual != "":
        contenido = texto_manual

    # Muestra el botón de "Predecir" si hay contenido
    if contenido:
        if st.button('Predecir', key='predict_button'):
            texto_limpio = limpiar_texto(contenido)
            prediccion = modelo.predict([texto_limpio])
            frase_prediccion = f"La gestante {gestante_nombre} SÍ necesita tratamiento antibiotico para prevenir sepsis por EGB." if prediccion[0] == 1 else f"La gestante {gestante_nombre} NO necesita tratamiento antibiotico para prevenir sepsis por EGB."
            
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
    st.sidebar.title('Introduce las credenciales para iniciar sesión:')
    usuario = st.sidebar.text_input("Usuario", key="user_input")
    contraseña = st.sidebar.text_input("Contraseña", type="password", key="password_input")
    if st.sidebar.button('Iniciar sesión', key='login_button'):
        if verificar_credenciales(usuario, contraseña):
            st.session_state['autenticado'] = True
            st.rerun()
        else:
            st.sidebar.error("Credenciales incorrectas.")
