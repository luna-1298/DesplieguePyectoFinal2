import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Despliegue de proyecto final", layout="wide")

# --- TÍTULO Y SUBTÍTULO REQUERIDOS ---
st.title("Despliegue de proyecto final")
st.subheader("Presentado por: Luna Tatiana Madroñero Jaimes y Juan José Restrepo salamanca")

st.markdown("""
Esta aplicación web integra todo el pipeline de análisis:
Carga de datos -> Filtrado de características -> Codificación -> Normalización -> Predicción mediante el modelo Boosting Optimizado.
""")

# --- CARGA DE RECURSOS (Modelo, Escalador y LabelEncoder) ---
SCALER_PATH = "min_max_scaler.joblib"
MODEL_PATH = "optimized_boosting_model.joblib"
LE_INTERNET_PATH = "label_encoder_internet.joblib"

@st.cache_resource
def load_resources():
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    le_internet = joblib.load(LE_INTERNET_PATH) if os.path.exists(LE_INTERNET_PATH) else None
    return scaler, model, le_internet

scaler, model, le_internet = load_resources()

if scaler is None or model is None or le_internet is None:
    st.warning("⚠️ Asegúrate de tener los archivos `min_max_scaler.joblib`, `optimized_boosting_model.joblib` y `label_encoder_internet.joblib` en el repositorio para que la predicción funcione correctamente.")

# --- SECCIÓN DE CARGA DE ARCHIVOS ---
st.subheader("📁 Carga de Datos Estudiantiles")
uploaded_file = st.file_uploader("Sube un archivo CSV con el formato de Student_Performance", type=["csv"])

# Permitir generación manual si no se sube un archivo
generate_random = st.checkbox("¿Generar un registro aleatorio de prueba si no hay archivo?")

df_input = None
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.success("Archivo cargado correctamente.")
    st.write("Vista previa de los datos cargados:")
    st.dataframe(df_input.head())
elif generate_random:
    # Datos ficticios de prueba
    data_placeholder = {
        'Age': [23.02], 'StudyTime_hours_week': [8.0], 'Failures': [2], 'Absences': [-3.0], 
        'FamilySupport': ['yes'], 'Internet': ['yes'], 'FreeTime': [5], 'GoOut': [2], 
        'Health': [5.0], 'MotherEducation': [2], 'FatherEducation': [1], 'TravelTime': [4]
    }
    df_input = pd.DataFrame(data_placeholder)
    st.info("Utilizando un registro de prueba autogenerado:")
    st.dataframe(df_input)

# --- PIPELINE DE PROCESAMIENTO Y PREDICCIÓN ---
if df_input is not None:
    if scaler is not None and model is not None and le_internet is not None:
        st.subheader("⚙️ Procesamiento, Codificación y Normalización")
        
        # 1. Copia y limpieza de variables no requeridas
        columns_to_drop = [
            'RecordID', 'Gender', 'FinalGrade', 'StudyTime', 
            'FullName', 'Phone', 'ZodiacSign', 'FavoriteColor', 'Hobby', 'FamilySupport'
        ]
        df_filtered = df_input.copy()
        existing_drops = [col for col in columns_to_drop if col in df_filtered.columns]
        df_filtered = df_filtered.drop(columns=existing_drops, errors='ignore')
        
        # 2. Aplicar LabelEncoder a la variable Internet
        if 'Internet' in df_filtered.columns:
            try:
                df_filtered['Internet'] = df_filtered['Internet'].astype(str)
                df_filtered['Internet'] = le_internet.transform(df_filtered['Internet'])
            except Exception as e:
                st.error(f"Error al codificar 'Internet': {e}")
            
        # 3. Normalización con el scaler
        columns_to_scale = [
            'Age', 'StudyTime_hours_week', 'Failures', 'Absences',
            'Internet', 'FreeTime', 'GoOut', 'Health',
            'MotherEducation', 'FatherEducation', 'TravelTime'
        ]
        available_to_scale = [col for col in columns_to_scale if col in df_filtered.columns]
        
        try:
            df_scaled = df_filtered.copy()
            df_scaled[available_to_scale] = scaler.transform(df_scaled[available_to_scale])
            
            st.write("Datos normalizados listos para el modelo:")
            st.dataframe(df_scaled)
            
            # 4. Alinear características para la predicción
            if hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_
                df_prediction = df_scaled[features]
            else:
                df_prediction = df_scaled
                
            # 5. Ejecutar la predicción
            prediction = model.predict(df_prediction)
            prediction_proba = model.predict_proba(df_prediction) if hasattr(model, 'predict_proba') else None
            
            # --- MOSTRAR RESULTADOS ---
            st.subheader("🎯 Resultados de la Predicción")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 1:
                    st.success("**Resultado Predicho: APROBADO (1)**")
                else:
                    st.error("**Resultado Predicho: REPROBADO (0)**")
                    
            with col2:
                if prediction_proba is not None:
                    st.metric("Probabilidad de Reprobar (Clase 0)", f"{prediction_proba[0][0]*100:.2f}%")
                    st.metric("Probabilidad de Aprobar (Clase 1)", f"{prediction_proba[0][1]*100:.2f}%")
                    
        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento o predicción: {e}")
    else:
        st.error("Faltan recursos esenciales para realizar la predicción (modelo, escalador o codificador de etiquetas).")
