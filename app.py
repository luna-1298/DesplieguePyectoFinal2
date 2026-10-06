import streamlit as st
import os
import sys
import traceback

st.set_page_config(page_title="Diagnóstico de Despliegue")
st.title("🛠️ Diagnóstico de Servidor y Archivos")

st.write("### 1. Entorno de Ejecución")
st.write(f"**Versión de Python:** {sys.version}")
st.write(f"**Directorio de trabajo actual:** `{os.getcwd()}`")

st.write("### 2. Archivos detectados en el repositorio")
try:
    files = os.listdir('.')
    st.write("Lista de archivos presentes en la raíz:")
    for f in files:
        if f.endswith('.joblib') or f.endswith('.py') or f.endswith('.txt'):
            st.write(f"- ✅ `{f}`")
        else:
            st.write(f"- `{f}`")
except Exception as e:
    st.error(f"Error al listar archivos: {e}")

st.write("### 3. Prueba de carga de dependencias críticas")
for lib_name in ['pandas', 'numpy', 'joblib', 'sklearn', 'xgboost', 'lightgbm']:
    try:
        __import__(lib_name)
        st.success(f"✅ `{lib_name}` importada correctamente.")
    except Exception as e:
        st.error(f"❌ Error al importar `{lib_name}`: {e}")
        st.code(traceback.format_exc())
