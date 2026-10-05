import os
import json
import pandas as pd
import streamlit as st

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Sistema Quant de Trading",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Tablero de Control Algorítmico y Auditoría de Activos")
st.markdown("---")

# Función para cargar los datos del escáner más reciente
@st.cache_data
def cargar_datos_escaneo():
    ruta_json = 'data/latest_market_scan.json'
    if os.path.exists(ruta_json):
        with open(ruta_json, 'r') as f:
            return json.load(f)
    return []

datos_escaneo = cargar_datos_escaneo()

if not datos_escaneo:
    st.warning("No se encontraron datos de escaneo recientes. Ejecuta primero `python src/scanner.py` en tu terminal.")
else:
    # Convertir a DataFrame para visualización tabular
    df_dashboard = pd.DataFrame(datos_escaneo)
    
    st.subheader("🚀 Resumen Ejecutivo del Mercado (Última Auditoría)")
    
    # Mostrar la tabla interactiva
    st.dataframe(df_dashboard, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 Detalle por Activo y Mandatos del Sistema")
    
    # Selector individual de activos para inspección profunda
    activo_seleccionado = st.selectbox("Selecciona un activo para ver sus detalles:", df_dashboard['Activo'].tolist())
    
    # Filtrar la información del activo seleccionado
    info_activo = df_dashboard[df_dashboard['Activo'] == activo_seleccionado].iloc[0]
    
    # Columnas visuales para las métricas clave
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Precio Actual (USD)", f"${info_activo['Cierre_USD']}")
    with col2:
        st.metric("Fuerza Tendencial (ADX)", info_activo['Fuerza_ADX'])
    with col3:
        st.metric("Índice RSI (14)", info_activo['RSI_14'])
    with col4:
        st.metric("Distancia Stop SAR (%)", f"{info_activo['SAR_Distancia_%']}%")
        
    st.markdown("---")
    
    # Órdenes y sugerencias de los diferentes modelos técnicos
    st.markdown(f"### 📋 Dictamen Técnico para: **{activo_seleccionado}**")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"**Fase Macro (Medias Móviles):** {info_activo['Fase_Mercado_Macro']}")
        st.info(f"**Diagnóstico MACD:** {info_activo['Diagnostico_MACD']}")
        st.info(f"**Mandato Flujo (OBV):** {info_activo['Mandato_OBV']}")
    with col_b:
        st.success(f"**Mandato Fuerza (ADX):** {info_activo['Mandato_ADX']}")
        st.success(f"**Mandato Trailing Stop (SAR):** {info_activo['Mandato_SAR']}")
        st.warning(f"**Sugerencia Global del Sistema:** {info_activo['Mandato_Global_Sugerido']}")

    st.markdown("---")
    st.markdown("*Proyecto Académico de Indicadores Técnicos y Automatización Cuantitativa.*")