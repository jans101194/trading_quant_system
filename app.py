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
    estrategias_obv, estrategias_macd, estrategias_adx, estrategias_rsi, estrategias_sar
)
from src.backtester import calcular_rendimientos, calcular_metricas

# ==========================================
# Configuración del Dashboard
# ==========================================
st.set_page_config(page_title="Terminal Quant | Institucional", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    div[data-testid="metric-container"] { background-color: #161a25; border: 1px solid #2b3040; padding: 1rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.4); }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 600; color: #00e676; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #8c9bb5; text-transform: uppercase; letter-spacing: 1px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; font-size: 1.1rem; font-weight: 600; }
    .stDataFrame { border: 1px solid #2b3040; border-radius: 5px; }
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
# CÁLCULOS DÍNAMICOS EN BACKEND
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
    df.dropna(inplace=True)
    df = calcular_medias_moviles(df, 'Close', mm_r, mm_l)
    df = calcular_bollinger(df, 'Close', bb_p, bb_s)
    df = calcular_macd(df, 'Close', m_r, m_l, m_s)
    df = calcular_estocastico(df, 'High', 'Low', 'Close', s_k, s_d)
    df = calcular_obv(df, 'Close', 'Volume')
    df = calcular_adx(df, 'High', 'Low', 'Close', a_p)
    df = calcular_rsi(df, 'Close', r_p)
    df = calcular_parabolic_sar(df, 'High', 'Low', sar_p, sar_m)
    return df

df_chart = obtener_datos_grafica(activo_seleccionado, periodo, mm_rapida, mm_lenta, bb_periodos, bb_std, macd_r, macd_l, macd_s, rsi_p, adx_p, sto_k, sto_d, sar_paso, sar_max)

# Integrar posiciones algorítmicas de forma obligatoria
if df_chart is not None and not df_chart.empty:
    df_chart = estrategias_medias_moviles(df_chart, mm_rapida, mm_lenta)
    df_chart = estrategias_bollinger(df_chart, bb_periodos)
    df_chart = estrategias_estocastico(df_chart)
    df_chart = estrategias_obv(df_chart, 20, 10, 30)
    df_chart = estrategias_macd(df_chart)
    df_chart = estrategias_adx(df_chart, 25)
    df_chart = estrategias_rsi(df_chart, rsi_p, 9)
    df_chart = estrategias_sar(df_chart, mm_lenta)

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
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'SMA_{mm_rapida}'], line=dict(color='#f6b26b', width=1.5), name=f'SMA {mm_rapida}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'SMA_{mm_lenta}'], line=dict(color='#6fa8dc', width=1.5), name=f'SMA {mm_lenta}'), row=1, col=1)
        if opcion_ema in overlays_seleccionados:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'EMA_{mm_rapida}'], line=dict(color='#00e676', width=1.5, dash='dash'), name=f'EMA {mm_rapida}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'EMA_{mm_lenta}'], line=dict(color='#e74c3c', width=1.5, dash='dash'), name=f'EMA {mm_lenta}'), row=1, col=1)
        if "Bandas de Bollinger" in overlays_seleccionados:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'UB_{bb_periodos}'], line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'), name='Banda Sup'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'LB_{bb_periodos}'], line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'), name='Banda Inf', fill='tonexty', fillcolor='rgba(255,255,255,0.03)'), row=1, col=1)
        if "Parabolic SAR" in overlays_seleccionados:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SAR'], mode='markers', marker=dict(color='#9b59b6', size=3), name='SAR'), row=1, col=1)

        current_row = 2
        for subplot in subplots_seleccionados:
            if subplot == "Volumen":
                colores_volumen = ['#00e676' if c >= o else '#ff4d4d' for c, o in zip(df_chart['Close'], df_chart['Open'])]
                fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colores_volumen, name='Volumen'), row=current_row, col=1)
            elif subplot == "MACD":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MACD_Line'], line=dict(color='#3498db', width=1.5), name='MACD'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Signal_Line'], line=dict(color='#ff9f43', width=1.5), name='Señal'), row=current_row, col=1)
                colores_hist = ['#00e676' if val >= 0 else '#ff4d4d' for val in df_chart['Histograma']]
                fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Histograma'], marker_color=colores_hist, name='Histograma'), row=current_row, col=1)
            elif subplot == "RSI":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[f'RSI_{rsi_p}'], line=dict(color='#f1c40f', width=1.5), name=f'RSI {rsi_p}'), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="#ff4d4d", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="#00e676", row=current_row, col=1)
            elif subplot == "OBV":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['OBV'], line=dict(color='#00cec9', width=1.5), name='OBV'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['OBV_SMA20'], line=dict(color='#fdcb6e', width=1.5, dash='dot'), name='SMA 20 (OBV)'), row=current_row, col=1)
            elif subplot == "ADX":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['ADX'], line=dict(color='#9b59b6', width=2), name=f'ADX {adx_p}'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['+DI'], line=dict(color='#00e676', width=1), name='+DI'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['-DI'], line=dict(color='#ff4d4d', width=1), name='-DI'), row=current_row, col=1)
                fig.add_hline(y=25, line_dash="dash", line_color="gray", row=current_row, col=1)
            elif subplot == "Estocástico":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['%K'], line=dict(color='#3498db', width=1.5), name='%K'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['%D'], line=dict(color='#ff9f43', width=1.5, dash='dot'), name='%D'), row=current_row, col=1)
                fig.add_hline(y=80, line_dash="dash", line_color="#ff4d4d", row=current_row, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="#00e676", row=current_row, col=1)
            current_row += 1

        fig.update_layout(height=500 + (200 * num_subplots), template='plotly_dark', plot_bgcolor='#12141c', paper_bgcolor='#12141c', showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="white")), margin=dict(l=10, r=10, t=30, b=10))
        for i in range(1, num_subplots + 2): fig.update_xaxes(rangeslider_visible=False, showgrid=True, gridcolor='#2b3040', row=i, col=1)
        st.plotly_chart(fig, use_container_width=True)

# ----------------- TAB 2: DATOS DINÁMICOS -----------------
with tab_datos:
    st.markdown(f"### Desglose Parametrizado y Señales en Tiempo Real ({activo_seleccionado})")
    if df_chart is not None and not df_chart.empty:
        ultima_fila = df_chart.iloc[-1]
        datos = {
            "Cierre USD": ultima_fila['Close'], "Volumen": ultima_fila['Volume'],
            f"SMA {mm_rapida}": ultima_fila[f'SMA_{mm_rapida}'], f"SMA {mm_lenta}": ultima_fila[f'SMA_{mm_lenta}'],
            f"EMA {mm_rapida}": ultima_fila[f'EMA_{mm_rapida}'], f"EMA {mm_lenta}": ultima_fila[f'EMA_{mm_lenta}'],
            f"RSI ({rsi_p})": ultima_fila[f'RSI_{rsi_p}'], f"ADX ({adx_p})": ultima_fila['ADX'],
            "MACD (Línea)": ultima_fila['MACD_Line'], "MACD (Señal)": ultima_fila['Signal_Line'],
            f"BB_UB ({bb_periodos})": ultima_fila[f'UB_{bb_periodos}'], f"BB_LB ({bb_periodos})": ultima_fila[f'LB_{bb_periodos}'],
            f"Estocástico %K": ultima_fila['%K'], "Parabolic SAR": ultima_fila['SAR'], "OBV": ultima_fila['OBV']
        }
        
        # Iterar sobre las columnas de posiciones (Pos_) e inyectar el mandato exacto al final
        for col in df_chart.columns:
            if col.startswith('Pos_'):
                val = ultima_fila[col]
                estado = "🟢 Largo / Compra" if val == 1.0 else "🔴 Corto / Venta" if val == -1.0 else "⚪ Cash / Neutral"
                datos[col.replace('Pos_', 'Señal ')] = estado

        df_dinamico = pd.DataFrame(list(datos.items()), columns=["Métrica / Estrategia", "Valor Registrado"])
        df_dinamico['Valor Registrado'] = df_dinamico['Valor Registrado'].apply(lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else str(x))
        st.dataframe(df_dinamico, use_container_width=True, hide_index=True)

# ----------------- TAB 3: BACKTESTING -----------------
with tab_backtest:
    st.markdown(f"### ⚙️ Motor Visual de Evaluación Institucional ({activo_seleccionado})")
    
    if df_chart is not None and not df_chart.empty:
        cols_posicion = [col for col in df_chart.columns if col.startswith('Pos_')]
        
        estrategia_visual = st.selectbox("🎯 Seleccione la Estrategia a Evaluar Sincronizada", cols_posicion, format_func=lambda x: x.replace('Pos_', ''))
        
        # Filtramos la curva de retornos sólo para la estrategia seleccionada y el Buy & Hold
        df_bt, cols_retornos = calcular_rendimientos(df_chart.copy(), 'Close', [estrategia_visual])
        nombre_estrategia = estrategia_visual.replace('Pos_', '')
        df_metrics = calcular_metricas(df_bt, cols_retornos, [nombre_estrategia])
        
        st.markdown(f"#### Análisis Sincronizado de Capital vs Entradas en `{nombre_estrategia}`")
        
        # Lienzo apilado compartido: Gráfico 1 (Precio + Señales) -> Gráfico 2 (Curva de Capital)
        fig_bt = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=("1. Diagnóstico Gráfico de Entradas y Salidas en Precio", "2. Evolución y Crecimiento Acumulado de Capital (Base 1.0)"),
            row_heights=[0.5, 0.5]
        )
        
        # --- Gráfico Superior: Acción del Precio y Puntos de Giro ---
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Close'], mode='lines', name='Precio Activo', line=dict(color='#8c9bb5', width=1.5)), row=1, col=1)
        
        # Lógica matemática precisa para atrapar los eventos en que el algoritmo cambia a un estado activo o neutro
        cambios = df_bt[estrategia_visual].diff()
        compras = df_bt[(cambios != 0) & (df_bt[estrategia_visual] == 1.0)]
        ventas_cortos = df_bt[(cambios != 0) & (df_bt[estrategia_visual] == -1.0)]
        salidas_cash = df_bt[(cambios != 0) & (df_bt[estrategia_visual] == 0.0)]
        
        fig_bt.add_trace(go.Scatter(x=compras.index, y=compras['Close'], mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00e676', line=dict(width=1, color='black')), name='Entrada (Long)'), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=ventas_cortos.index, y=ventas_cortos['Close'], mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff4d4d', line=dict(width=1, color='black')), name='Venta (Short)'), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=salidas_cash.index, y=salidas_cash['Close'], mode='markers', marker=dict(symbol='x', size=10, color='#f1c40f'), name='Salida a Cash (Filtro)'), row=1, col=1)

        # --- Gráfico Inferior: Curvas de Equity Comparativas ---
        ret_col = cols_retornos[1]
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Cum_Retorno_Mercado'], mode='lines', name='Buy & Hold (Mercado)', line=dict(color='#00e676', width=2.5)), row=2, col=1)
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt[f'Cum_{ret_col}'], mode='lines', name=f'Rendimiento {nombre_estrategia}', line=dict(color='#e74c3c', width=2)), row=2, col=1)
        
        fig_bt.update_layout(height=750, template='plotly_dark', plot_bgcolor='#12141c', paper_bgcolor='#12141c', margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_bt.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig_bt.update_xaxes(rangeslider_visible=False, row=2, col=1)
        fig_bt.update_yaxes(title_text="Precio USD", row=1, col=1)
        fig_bt.update_yaxes(title_text="Capital Base 1", row=2, col=1)

        st.plotly_chart(fig_bt, use_container_width=True)

        st.markdown("#### Tabla de Desempeño Cuantitativo Institucional")
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)