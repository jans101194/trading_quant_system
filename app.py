import os
import json
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importar ecosistema completo
from src.indicators import (
    calcular_medias_moviles, calcular_bollinger, calcular_macd,
    calcular_estocastico, calcular_obv, calcular_adx,
    calcular_rsi, calcular_parabolic_sar
)
from src.strategies import (
    estrategias_medias_moviles, estrategias_bollinger, estrategias_estocastico,
    estrategias_obv, estrategias_macd, estrategias_adx, estrategias_rsi, estrategias_sar,
    estrategias_conjuntas
)
from src.backtester import calcular_rendimientos, calcular_metricas

# ==========================================
# Configuración del Dashboard
# ==========================================
st.set_page_config(page_title="Terminal Quant | Institucional", layout="wide", initial_sidebar_state="expanded")

# CSS modificado para un tema claro y profesional
st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    div[data-testid="metric-container"] { background-color: #ffffff; border: 1px solid #e0e6ed; padding: 1rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 600; color: #204a87; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #5c677d; text-transform: uppercase; letter-spacing: 1px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; font-size: 1.1rem; font-weight: 600; color: #333333; }
    .stDataFrame { border: 1px solid #e0e6ed; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_scanner_data():
    if os.path.exists('data/latest_market_scan.json'):
        try:
            with open('data/latest_market_scan.json', 'r') as f: return json.load(f)
        except: return []
    return []

scanner_data = load_scanner_data()
df_scanner = pd.DataFrame(scanner_data) if scanner_data else pd.DataFrame()

# ==========================================
# Barra Lateral (Sidebar)
# ==========================================
st.sidebar.title("📊 Terminal Cuantitativa")
st.sidebar.markdown("---")

activos_disponibles = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "BTC-USD"]
if not df_scanner.empty and "Activo" in df_scanner.columns:
    activos_disponibles = df_scanner["Activo"].tolist()

activo_seleccionado = st.sidebar.selectbox("🎯 Selección de Activo", activos_disponibles)
periodo = st.sidebar.select_slider("⏱ Período Histórico", options=["6mo", "1y", "2y", "4y"], value="1y")

# --- PARAMETRIZACIÓN DINÁMICA ---
with st.sidebar.expander("🎛️ Parámetros de Indicadores", expanded=False):
    st.markdown("**Medias Móviles**")
    mm_rapida = st.number_input("Rápida", 5, 200, 20)
    mm_lenta = st.number_input("Lenta", 10, 300, 50)
    
    st.markdown("**Bandas de Bollinger**")
    bb_periodos = st.number_input("Periodos (BB)", 5, 100, 20)
    bb_std = st.number_input("Desviación", 1.0, 5.0, 2.0, step=0.1)
    
    st.markdown("**MACD**")
    macd_r = st.number_input("Rápida (MACD)", 2, 50, 12)
    macd_l = st.number_input("Lenta (MACD)", 5, 100, 26)
    macd_s = st.number_input("Señal (MACD)", 2, 50, 9)
    
    st.markdown("**RSI & ADX**")
    rsi_p = st.number_input("Periodos RSI", 2, 100, 14)
    adx_p = st.number_input("Periodos ADX", 2, 100, 14)
    
    st.markdown("**Estocástico**")
    sto_k = st.number_input("%K", 2, 50, 14)
    sto_d = st.number_input("%D", 2, 50, 3)
    
    st.markdown("**Parabolic SAR**")
    sar_paso = st.number_input("Paso SAR", 0.001, 0.1, 0.02, format="%.3f")
    sar_max = st.number_input("Máx SAR", 0.05, 1.0, 0.2, format="%.2f")

st.sidebar.markdown("---")
opcion_sma = f"SMA ({mm_rapida}, {mm_lenta})"
opcion_ema = f"EMA ({mm_rapida}, {mm_lenta})"

st.sidebar.markdown("**Capas del Gráfico Principal**")
overlays_seleccionados = st.sidebar.multiselect("Superposiciones Técnicas", [opcion_sma, opcion_ema, "Bandas de Bollinger", "Parabolic SAR"], default=[opcion_ema, "Bandas de Bollinger"], label_visibility="collapsed")

st.sidebar.markdown("**Paneles Inferiores**")
subplots_seleccionados = st.sidebar.multiselect("Osciladores", ["Volumen", "MACD", "RSI", "OBV", "ADX", "Estocástico"], default=["Volumen", "MACD"], label_visibility="collapsed")

# ==========================================
# CÁLCULOS DINÁMICOS EN BACKEND
# ==========================================
@st.cache_data(show_spinner="Procesando datos...")
def obtener_datos_grafica(ticker, period, mm_r, mm_l, bb_p, bb_s, m_r, m_l, m_s, r_p, a_p, s_k, s_d, sar_p, sar_m):
    df = yf.download(ticker, period=period, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1) if ticker in df.columns.get_level_values(1) else df.copy()
        df.columns = df.columns.get_level_values(0)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = calcular_medias_moviles(df, 'Close', mm_r, mm_l)
    df = calcular_bollinger(df, 'Close', bb_p, bb_s)
    df = calcular_macd(df, 'Close', m_r, m_l, m_s)
    df = calcular_estocastico(df, 'High', 'Low', 'Close', s_k, s_d)
    df = calcular_obv(df, 'Close', 'Volume')
    df = calcular_adx(df, 'High', 'Low', 'Close', a_p)
    df = calcular_rsi(df, 'Close', r_p)
    df = calcular_parabolic_sar(df, 'High', 'Low', sar_p, sar_m)
    df.dropna(inplace=True)
    return df

df_chart = obtener_datos_grafica(activo_seleccionado, periodo, mm_rapida, mm_lenta, bb_periodos, bb_std, macd_r, macd_l, macd_s, rsi_p, adx_p, sto_k, sto_d, sar_paso, sar_max)

if df_chart is not None and not df_chart.empty:
    df_chart = estrategias_medias_moviles(df_chart, mm_r=mm_rapida, mm_l=mm_lenta)
    df_chart = estrategias_bollinger(df_chart, periodos=bb_periodos)
    df_chart = estrategias_estocastico(df_chart)
    df_chart = estrategias_obv(df_chart, sma_obv=20, ema_obv=10, sma2_obv=30, sma_precio=mm_rapida)
    df_chart = estrategias_macd(df_chart)
    df_chart = estrategias_adx(df_chart, umbral=25)
    df_chart = estrategias_rsi(df_chart, rsi_p=rsi_p, sma_rsi=9)
    df_chart = estrategias_sar(df_chart, ema_l=mm_lenta)
    df_chart = estrategias_conjuntas(df_chart, ema_l=mm_lenta, bb_p=bb_periodos, rsi_p=rsi_p, adx_u=25)

# ==========================================
# VISTA PRINCIPAL (3 PESTAÑAS)
# ==========================================
tab_grafico, tab_datos, tab_backtest = st.tabs(["📈 Análisis Gráfico", "📋 Desglose Técnico", "⚙️ Evaluación y Backtesting"])

# ----------------- TAB 1: GRÁFICO -----------------
with tab_grafico:
    if not df_scanner.empty:
        df_activo = df_scanner[df_scanner["Activo"] == activo_seleccionado]
        if not df_activo.empty:
            data_activo = df_activo.iloc[0]
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            mandato = data_activo.get('Mandato_Global_Sugerido', 'N/A')
            color_mandato = "🔴" if "CORT" in mandato.upper() or "VEND" in mandato.upper() else ("🟢" if "LARG" in mandato.upper() or "COMP" in mandato.upper() else "⚪")

            kpi1.metric("Último Cierre", f"${data_activo.get('Cierre_USD', 0)}")
            kpi2.metric("RSI (Scanner)", data_activo.get('RSI_14', 0))
            kpi3.metric("ADX (Scanner)", data_activo.get('Fuerza_ADX', 0))
            kpi4.metric("Señal MACD", data_activo.get('Diagnostico_MACD', 'N/A'))
            kpi5.metric("Mandato", f"{color_mandato} {mandato}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if df_chart is not None and not df_chart.empty:
        num_subplots = len(subplots_seleccionados)
        row_heights = [0.5] + [(0.5 / num_subplots)] * num_subplots if num_subplots > 0 else [1.0]
        fig = make_subplots(rows=num_subplots+1, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=[f'Acción de Precio'] + subplots_seleccionados, row_heights=row_heights)

        fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Precio'), row=1, col=1)
        
        if opcion_sma in overlays_seleccionados:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'SMA_{mm_rapida}'], line=dict(color='#e67e22', width=1.5), name=f'SMA {mm_rapida}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'SMA_{mm_lenta}'], line=dict(color='#2980b9', width=1.5), name=f'SMA {mm_lenta}'), row=1, col=1)
        if opcion_ema in overlays_seleccionados:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'EMA_{mm_rapida}'], line=dict(color='#27ae60', width=1.5, dash='dash'), name=f'EMA {mm_rapida}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'EMA_{mm_lenta}'], line=dict(color='#c0392b', width=1.5, dash='dash'), name=f'EMA {mm_lenta}'), row=1, col=1)
        if "Bandas de Bollinger" in overlays_seleccionados:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'UB_{bb_periodos}'], line=dict(color='rgba(0,0,0,0.2)', width=1, dash='dot'), name='Banda Sup'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'LB_{bb_periodos}'], line=dict(color='rgba(0,0,0,0.2)', width=1, dash='dot'), name='Banda Inf', fill='tonexty', fillcolor='rgba(0,0,0,0.05)'), row=1, col=1)
        if "Parabolic SAR" in overlays_seleccionados:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SAR'], mode='markers', marker=dict(color='#8e44ad', size=3), name='SAR'), row=1, col=1)

        current_row = 2
        for subplot in subplots_seleccionados:
            if subplot == "Volumen":
                colores_volumen = ['#2ecc71' if c >= o else '#e74c3c' for c, o in zip(df_chart['Close'], df_chart['Open'])]
                fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colores_volumen, name='Volumen'), row=current_row, col=1)
            elif subplot == "MACD":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MACD_Line'], line=dict(color='#2980b9', width=1.5), name='MACD'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Signal_Line'], line=dict(color='#d35400', width=1.5), name='Señal'), row=current_row, col=1)
                colores_hist = ['#2ecc71' if val >= 0 else '#e74c3c' for val in df_chart['Histograma']]
                fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Histograma'], marker_color=colores_hist, name='Histograma'), row=current_row, col=1)
            elif subplot == "RSI":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'RSI_{rsi_p}'], line=dict(color='#d35400', width=1.5), name=f'RSI {rsi_p}'), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="#e74c3c", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="#2ecc71", row=current_row, col=1)
            elif subplot == "OBV":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['OBV'], line=dict(color='#16a085', width=1.5), name='OBV'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['OBV_SMA20'], line=dict(color='#f39c12', width=1.5, dash='dot'), name='SMA 20 (OBV)'), row=current_row, col=1)
            elif subplot == "ADX":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['ADX'], line=dict(color='#8e44ad', width=2), name=f'ADX {adx_p}'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['+DI'], line=dict(color='#27ae60', width=1), name='+DI'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['-DI'], line=dict(color='#c0392b', width=1), name='-DI'), row=current_row, col=1)
                fig.add_hline(y=25, line_dash="dash", line_color="#7f8c8d", row=current_row, col=1)
            elif subplot == "Estocástico":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['%K'], line=dict(color='#2980b9', width=1.5), name='%K'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['%D'], line=dict(color='#d35400', width=1.5, dash='dot'), name='%D'), row=current_row, col=1)
                fig.add_hline(y=80, line_dash="dash", line_color="#e74c3c", row=current_row, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="#2ecc71", row=current_row, col=1)
            current_row += 1

        fig.update_layout(
            height=500 + (200 * num_subplots),
            template='plotly_white',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#333333")),
            margin=dict(l=10, r=10, t=30, b=10)
        )
        for i in range(1, num_subplots + 2):
            fig.update_xaxes(rangeslider_visible=False, showgrid=True, gridcolor='#e0e6ed', row=i, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='#e0e6ed', row=i, col=1)
            
        st.plotly_chart(fig, use_container_width=True)

# ----------------- TAB 2: DATOS DINÁMICOS -----------------
with tab_datos:
    st.markdown(f"### 📋 Matriz de Desglose Técnico ({activo_seleccionado})")
    st.markdown("Evaluación algorítmica y paramétrica en el último corte de mercado (T=0).")
    
    if df_chart is not None and not df_chart.empty:
        ultima_fila = df_chart.iloc[-1]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cierre Actual", f"${ultima_fila['Close']:,.2f}")
        c2.metric("Volumen Negociado", f"{ultima_fila['Volume']:,.0f}")
        c3.metric(f"RSI ({rsi_p})", f"{ultima_fila[f'RSI_{rsi_p}']:.2f}")
        c4.metric(f"Fuerza ADX ({adx_p})", f"{ultima_fila['ADX']:.2f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Análisis Multiestrategia por Familia de Indicador")
        
        filas_desglose_individual = []
        filas_desglose_conjunta = []
        
        for col in df_chart.columns:
            if col.startswith('Pos_'):
                val = ultima_fila[col]
                estado = "🟢 LARGO / COMPRA" if val == 1.0 else "🔴 CORTO / VENTA" if val == -1.0 else "⚪ NEUTRAL / ESPERA"
                
                familia = ""
                valores = ""
                nombre_estrategia = col.replace('Pos_', '')
                
                if 'Pos_CONJ_' in col:
                    familia = "Sinergia Conjunta"
                    if 'E1' in col:
                        valores = f"Cierre: {ultima_fila['Close']:.2f} | EMA 50: {ultima_fila[f'EMA_{mm_lenta}']:.2f} | MACD: {ultima_fila['MACD_Line']:.2f}"
                    elif 'E2' in col:
                        valores = f"BB Low: {ultima_fila[f'LB_{bb_periodos}']:.2f} | BB Up: {ultima_fila[f'UB_{bb_periodos}']:.2f} | RSI: {ultima_fila[f'RSI_{rsi_p}']:.2f}"
                    elif 'E3' in col:
                        valores = f"ADX: {ultima_fila['ADX']:.2f} | EMA: {ultima_fila[f'EMA_{mm_lenta}']:.2f} | %K: {ultima_fila['%K']:.2f}"
                elif 'MM_' in col:
                    familia = "Medias Móviles"
                    valores = f"Cierre: {ultima_fila['Close']:.2f} | Rápida: {ultima_fila[f'EMA_{mm_rapida}']:.2f} | Lenta: {ultima_fila[f'EMA_{mm_lenta}']:.2f}"
                elif 'BB_' in col:
                    familia = "Bandas Bollinger"
                    valores = f"UB: {ultima_fila[f'UB_{bb_periodos}']:.2f} | Cierre: {ultima_fila['Close']:.2f} | LB: {ultima_fila[f'LB_{bb_periodos}']:.2f}"
                elif 'STOCH_' in col:
                    familia = "Estocástico"
                    valores = f"%K: {ultima_fila['%K']:.2f} | %D: {ultima_fila['%D']:.2f}"
                elif 'OBV_' in col:
                    familia = "On-Balance Vol"
                    valores = f"OBV: {ultima_fila['OBV']:,.0f} | SMA20: {ultima_fila['OBV_SMA20']:,.0f}"
                elif 'MACD_' in col:
                    familia = "MACD"
                    valores = f"MACD: {ultima_fila['MACD_Line']:.2f} | Señal: {ultima_fila['Signal_Line']:.2f} | Hist: {ultima_fila['Histograma']:.2f}"
                elif 'ADX_' in col:
                    familia = "ADX / DMI"
                    valores = f"ADX: {ultima_fila['ADX']:.2f} | +DI: {ultima_fila['+DI']:.2f} | -DI: {ultima_fila['-DI']:.2f}"
                elif 'RSI_' in col:
                    familia = "RSI"
                    valores = f"RSI: {ultima_fila[f'RSI_{rsi_p}']:.2f} | M.A: {ultima_fila.get(f'SMA9_del_RSI', 0):.2f}"
                elif 'SAR_' in col:
                    familia = "Parabolic SAR"
                    valores = f"SAR: {ultima_fila['SAR']:.2f} | Cierre: {ultima_fila['Close']:.2f}"
                
                fila_obj = {
                    "Familia Técnica": familia,
                    "Variante / Estrategia Evaluada": nombre_estrategia.replace('_', ' '),
                    "Lecturas Clave del Algoritmo": valores,
                    "Mandato (Señal)": estado
                }
                
                if 'Pos_CONJ_' in col:
                    filas_desglose_conjunta.append(fila_obj)
                else:
                    filas_desglose_individual.append(fila_obj)
        
        column_cfg = {
            "Familia Técnica": st.column_config.TextColumn("Familia Técnica", width="medium"),
            "Variante / Estrategia Evaluada": st.column_config.TextColumn("Variante / Estrategia Evaluada", width="medium"),
            "Lecturas Clave del Algoritmo": st.column_config.TextColumn("Lecturas Clave del Algoritmo", width="large"),
            "Mandato (Señal)": st.column_config.TextColumn("Mandato (Señal)", width="medium")
        }

        df_dinamico = pd.DataFrame(filas_desglose_individual)
        st.dataframe(df_dinamico, use_container_width=True, hide_index=True, column_config=column_cfg)

        st.markdown("#### 🔗 Sinergia Cuantitativa (Sistemas de Confirmación Múltiple)")
        df_conjunto = pd.DataFrame(filas_desglose_conjunta)
        st.dataframe(df_conjunto, use_container_width=True, hide_index=True, column_config=column_cfg)

# ----------------- TAB 3: BACKTESTING -----------------
with tab_backtest:
    st.markdown(f"### ⚙️ Motor Visual de Evaluación Institucional ({activo_seleccionado})")
    
    if df_chart is not None and not df_chart.empty:
        cols_posicion = [col for col in df_chart.columns if col.startswith('Pos_')]
        
        familias = {
            "Medias Móviles": "Pos_MM_",
            "Bandas de Bollinger": "Pos_BB_",
            "Oscilador Estocástico": "Pos_STOCH_",
            "On-Balance Volume (OBV)": "Pos_OBV_",
            "MACD": "Pos_MACD_",
            "ADX": "Pos_ADX_",
            "RSI": "Pos_RSI_",
            "Parabolic SAR": "Pos_SAR_",
            "Estrategias Conjuntas": "Pos_CONJ_"
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            familia_seleccionada = st.selectbox("📊 Seleccione la Familia de Indicadores", list(familias.keys()))
        
        prefijo = familias[familia_seleccionada]
        estrategias_disponibles = [col for col in cols_posicion if col.startswith(prefijo)]
        
        with col2:
            estrategia_visual = st.selectbox(
                "🎯 Seleccione la Estrategia a Evaluar Sincronizada",
                estrategias_disponibles,
                format_func=lambda x: x.replace(prefijo, '').replace('_', ' ')
            )
        
        df_bt, cols_retornos = calcular_rendimientos(df_chart.copy(), 'Close', [estrategia_visual])
        nombre_estrategia = estrategia_visual.replace('Pos_', '')
        df_metrics = calcular_metricas(df_bt, cols_retornos, [nombre_estrategia])
        
        st.markdown(f"#### Análisis Sincronizado de Capital vs Entradas en `{nombre_estrategia}`")
        
        es_sistema_institucional = familia_seleccionada == "Estrategias Conjuntas" and "E3" in estrategia_visual
        requiere_subplot = familia_seleccionada in ["Oscilador Estocástico", "On-Balance Volume (OBV)", "MACD", "ADX", "RSI", "Estrategias Conjuntas"]
        
        if es_sistema_institucional:
            fig_bt = make_subplots(
                rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                subplot_titles=(
                    f"1. Entradas y Salidas: {nombre_estrategia}",
                    "2. Filtro Direccional (ADX)",
                    "3. Gatillo Estocástico",
                    "4. Evolución y Crecimiento Acumulado de Capital (Base 1.0)"
                ),
                row_heights=[0.4, 0.2, 0.2, 0.3]
            )
            row_equity = 4
            altura_grafico = 1000
        elif requiere_subplot:
            fig_bt = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                subplot_titles=(
                    f"1. Entradas y Salidas: {nombre_estrategia}",
                    f"2. Indicador Técnico: {familia_seleccionada}",
                    "3. Evolución y Crecimiento Acumulado de Capital (Base 1.0)"
                ),
                row_heights=[0.4, 0.3, 0.3]
            )
            row_equity = 3
            altura_grafico = 900
        else:
            fig_bt = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                subplot_titles=(
                    f"1. Entradas y Salidas + Capas de {familia_seleccionada}",
                    "2. Evolución y Crecimiento Acumulado de Capital (Base 1.0)"
                ),
                row_heights=[0.5, 0.5]
            )
            row_equity = 2
            altura_grafico = 750

        # --- Fila 1: Acción del Precio ---
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Close'], mode='lines', name='Precio Activo', line=dict(color='#5c677d', width=1.5)), row=1, col=1)
        
        cambios = df_bt[estrategia_visual].diff()
        compras = df_bt[(cambios != 0) & (df_bt[estrategia_visual] == 1.0)]
        ventas_cortos = df_bt[(cambios != 0) & (df_bt[estrategia_visual] == -1.0)]
        salidas_cash = df_bt[(cambios != 0) & (df_bt[estrategia_visual] == 0.0)]
        
        fig_bt.add_trace(go.Scatter(x=compras.index, y=compras['Close'], mode='markers', marker=dict(symbol='triangle-up', size=14, color='#2ecc71', line=dict(width=1, color='black')), name='Entrada (Long)'), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=ventas_cortos.index, y=ventas_cortos['Close'], mode='markers', marker=dict(symbol='triangle-down', size=14, color='#e74c3c', line=dict(width=1, color='black')), name='Venta (Short)'), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=salidas_cash.index, y=salidas_cash['Close'], mode='markers', marker=dict(symbol='x', size=10, color='#f39c12'), name='Salida a Cash (Filtro)'), row=1, col=1)

        # --- Indicadores Técnicos ---
        if familia_seleccionada == "Estrategias Conjuntas":
            if "E1" in estrategia_visual:
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'EMA_{mm_lenta}'], line=dict(color='#e67e22', width=1.5), name=f'EMA {mm_lenta}'), row=1, col=1)
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['MACD_Line'], line=dict(color='#2980b9', width=1.5), name='MACD'), row=2, col=1)
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Signal_Line'], line=dict(color='#d35400', width=1.5), name='Señal'), row=2, col=1)
                colores_hist = ['#2ecc71' if val >= 0 else '#e74c3c' for val in df_bt['Histograma']]
                fig_bt.add_trace(go.Bar(x=df_bt.index, y=df_bt['Histograma'], marker_color=colores_hist, name='Histograma'), row=2, col=1)
            elif "E2" in estrategia_visual:
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'UB_{bb_periodos}'], line=dict(color='rgba(0,0,0,0.2)', width=1, dash='dot'), name='Banda Sup'), row=1, col=1)
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'LB_{bb_periodos}'], line=dict(color='rgba(0,0,0,0.2)', width=1, dash='dot'), name='Banda Inf', fill='tonexty', fillcolor='rgba(0,0,0,0.05)'), row=1, col=1)
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'RSI_{rsi_p}'], line=dict(color='#8e44ad', width=1.5), name=f'RSI {rsi_p}'), row=2, col=1)
                fig_bt.add_hline(y=70, line_dash="dash", line_color="#e74c3c", row=2, col=1)
                fig_bt.add_hline(y=30, line_dash="dash", line_color="#2ecc71", row=2, col=1)
            elif "E3" in estrategia_visual:
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'EMA_{mm_lenta}'], line=dict(color='#8e44ad', width=1.5), name=f'EMA {mm_lenta}'), row=1, col=1)
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['ADX'], line=dict(color='black', width=2), name=f'ADX {adx_p}'), row=2, col=1)
                fig_bt.add_hline(y=25, line_dash="dash", line_color="gray", row=2, col=1)
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['%K'], line=dict(color='#3498db', width=1.5), name='%K'), row=3, col=1)
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['%D'], line=dict(color='#e74c3c', width=1.5, dash='dot'), name='%D'), row=3, col=1)
                fig_bt.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
                fig_bt.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)

        elif familia_seleccionada == "Medias Móviles":
            if "SMA" in estrategia_visual:
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'SMA_{mm_rapida}'], line=dict(color='#e67e22', width=1.5), name=f'SMA {mm_rapida}'), row=1, col=1)
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'SMA_{mm_lenta}'], line=dict(color='#2980b9', width=1.5), name=f'SMA {mm_lenta}'), row=1, col=1)
            else:
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'EMA_{mm_rapida}'], line=dict(color='#27ae60', width=1.5), name=f'EMA {mm_rapida}'), row=1, col=1)
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'EMA_{mm_lenta}'], line=dict(color='#c0392b', width=1.5), name=f'EMA {mm_lenta}'), row=1, col=1)

        elif familia_seleccionada == "Bandas de Bollinger":
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'UB_{bb_periodos}'], line=dict(color='rgba(0,0,0,0.2)', width=1, dash='dot'), name='Banda Sup'), row=1, col=1)
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'MB_{bb_periodos}'], line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dot'), name='Media Central'), row=1, col=1)
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'LB_{bb_periodos}'], line=dict(color='rgba(0,0,0,0.2)', width=1, dash='dot'), name='Banda Inf', fill='tonexty', fillcolor='rgba(0,0,0,0.05)'), row=1, col=1)

        elif familia_seleccionada == "Parabolic SAR":
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['SAR'], mode='markers', marker=dict(color='#8e44ad', size=3), name='SAR'), row=1, col=1)
            if "Filtro_EMA" in estrategia_visual:
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'EMA_{mm_lenta}'], line=dict(color='#2980b9', width=1.5), name=f'EMA {mm_lenta}'), row=1, col=1)

        elif familia_seleccionada == "MACD":
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['MACD_Line'], line=dict(color='#2980b9', width=1.5), name='MACD'), row=2, col=1)
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Signal_Line'], line=dict(color='#d35400', width=1.5), name='Señal'), row=2, col=1)
            colores_hist = ['#2ecc71' if val >= 0 else '#e74c3c' for val in df_bt['Histograma']]
            fig_bt.add_trace(go.Bar(x=df_bt.index, y=df_bt['Histograma'], marker_color=colores_hist, name='Histograma'), row=2, col=1)

        elif familia_seleccionada == "RSI":
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'RSI_{rsi_p}'], line=dict(color='#d35400', width=1.5), name=f'RSI {rsi_p}'), row=2, col=1)
            if "Gatillo" in estrategia_visual:
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['SMA9_del_RSI'], line=dict(color='#2980b9', width=1.5, dash='dot'), name='SMA 9 RSI'), row=2, col=1)
            fig_bt.add_hline(y=70, line_dash="dash", line_color="#e74c3c", row=2, col=1)
            fig_bt.add_hline(y=50, line_dash="dash", line_color="#bdc3c7", row=2, col=1)
            fig_bt.add_hline(y=30, line_dash="dash", line_color="#2ecc71", row=2, col=1)

        elif familia_seleccionada == "Oscilador Estocástico":
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['%K'], line=dict(color='#2980b9', width=1.5), name='%K'), row=2, col=1)
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['%D'], line=dict(color='#d35400', width=1.5, dash='dot'), name='%D'), row=2, col=1)
            fig_bt.add_hline(y=80, line_dash="dash", line_color="#e74c3c", row=2, col=1)
            fig_bt.add_hline(y=20, line_dash="dash", line_color="#2ecc71", row=2, col=1)

        elif familia_seleccionada == "ADX":
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['ADX'], line=dict(color='#8e44ad', width=2), name=f'ADX {adx_p}'), row=2, col=1)
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['+DI'], line=dict(color='#27ae60', width=1), name='+DI'), row=2, col=1)
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['-DI'], line=dict(color='#c0392b', width=1), name='-DI'), row=2, col=1)
            fig_bt.add_hline(y=25, line_dash="dash", line_color="#7f8c8d", row=2, col=1)

        elif familia_seleccionada == "On-Balance Volume (OBV)":
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['OBV'], line=dict(color='#16a085', width=1.5), name='OBV'), row=2, col=1)
            if "Cruce_Medias" in estrategia_visual:
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['OBV_EMA10'], line=dict(color='#d35400', width=1.5), name='OBV EMA 10'), row=2, col=1)
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['OBV_SMA30'], line=dict(color='#2980b9', width=1.5, dash='dot'), name='OBV SMA 30'), row=2, col=1)
            else:
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['OBV_SMA20'], line=dict(color='#f39c12', width=1.5, dash='dot'), name='OBV SMA 20'), row=2, col=1)

        # --- Gráfico de Capital ---
        ret_col = cols_retornos[1]
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Cum_Retorno_Mercado'], mode='lines', name='Buy & Hold (Mercado)', line=dict(color='#2ecc71', width=2.5)), row=row_equity, col=1)
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'Cum_{ret_col}'], mode='lines', name=f'Rendimiento {nombre_estrategia}', line=dict(color='#c0392b', width=2)), row=row_equity, col=1)
        
        fig_bt.update_layout(
            height=altura_grafico,
            template='plotly_white',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#333333"))
        )
        
        filas_totales = 4 if es_sistema_institucional else (3 if requiere_subplot else 2)
        for i in range(1, filas_totales + 1):
            fig_bt.update_xaxes(rangeslider_visible=False, showgrid=True, gridcolor='#e0e6ed', row=i, col=1)
            fig_bt.update_yaxes(showgrid=True, gridcolor='#e0e6ed', row=i, col=1)
        
        fig_bt.update_yaxes(title_text="Precio USD", row=1, col=1)
        fig_bt.update_yaxes(title_text="Capital Base 1", row=row_equity, col=1)

        st.plotly_chart(fig_bt, use_container_width=True)

        st.markdown("#### Tabla de Desempeño Cuantitativo Institucional")
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)