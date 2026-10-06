import os
import json
import datetime
import yfinance as yf
import pandas as pd
import numpy as np

# Importamos nuestro motor matemático
from indicators import (
    calcular_medias_moviles, calcular_bollinger, calcular_estocastico,
    calcular_obv, calcular_macd, calcular_adx, calcular_rsi, calcular_parabolic_sar
)

def auditar_activo(ticker):
    """
    Microservicio de evaluación técnica integral.
    Descarga datos, calcula indicadores y emite un mandato algorítmico consolidado.
    """
    try:
        # 1. Extracción de datos en modo silencioso (6 meses son suficientes para convergencia EMA/RMA)
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty:
            return {"Activo": ticker, "Error": "Sin respuesta de datos."}

        # Adaptación robusta para MultiIndex en nuevas versiones de yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, level=1, axis=1) if ticker in df.columns.get_level_values(1) else df.copy()
            df.columns = df.columns.get_level_values(0)

        # Aseguramos formato numérico
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2. Ingesta Matemática (Cálculo Vectorizado sobre toda la muestra)
        df = calcular_medias_moviles(df, col_precio='Close')
        df = calcular_bollinger(df, col_precio='Close')
        df = calcular_estocastico(df, col_high='High', col_low='Low', col_close='Close')
        df = calcular_obv(df, col_precio='Close', col_volumen='Volume')
        df = calcular_macd(df, col_precio='Close')
        df = calcular_adx(df, col_high='High', col_low='Low', col_close='Close')
        df = calcular_rsi(df, col_precio='Close')
        df = calcular_parabolic_sar(df, col_high='High', col_low='Low')

        # 3. Limpieza de valores nulos (Periodo de calentamiento)
        df.dropna(inplace=True)

        # Verificamos si quedaron suficientes datos después de descartar el calentamiento
        if len(df) < 10:
            return {"Activo": ticker, "Error": "Datos insuficientes tras calentamiento."}

        # 4. Telemetría de Frontera (T=0 y T-1)
        cierre_actual = float(df['Close'].iloc[-1])
        fecha_corte = df.index[-1].strftime('%Y-%m-%d')

        # -- Medias Móviles (Doble Cruce EMA 20/50) --
        r_actual, l_actual = float(df['EMA_20'].iloc[-1]), float(df['EMA_50'].iloc[-1])
        r_previa, l_previa = float(df['EMA_20'].iloc[-2]), float(df['EMA_50'].iloc[-2])
        mm_estado, mm_orden = "Zona de Transición", "ESPERAR"
        
        if (r_previa < l_previa) and (r_actual > l_actual):
            mm_estado, mm_orden = "Ruptura Alcista", "Entrar Largo"
        elif (r_previa > l_previa) and (r_actual < l_actual):
            mm_estado, mm_orden = "Fallo Bajista", "Entrar Corto"
        elif r_actual > l_actual:
            distancia = ((r_actual - l_actual) / l_actual) * 100
            if distancia > 5.0:
                mm_estado, mm_orden = f"Alcista (Sobreextendida: +{distancia:.1f}%)", "Hold Largo"
            else:
                mm_estado, mm_orden = "Tendencia Alcista", "Hold Largo"
        else:
            mm_estado, mm_orden = "Bajista Dominante", "Hold Corto"

        # -- Bandas de Bollinger (Squeeze & Breakout) --
        pct_b_actual = float(df['Pct_B'].iloc[-1])
        bw_actual = float(df['Bandwidth'].iloc[-1])
        bw_historico = df['Bandwidth'].values
        umbral_squeeze = np.percentile(bw_historico, 20)
        es_squeeze = bw_actual < umbral_squeeze
        bb_estado, bb_orden = "Normal de Volatilidad", "ESPERAR"

        if es_squeeze:
            bb_estado = "Estrangulamiento Extremo"
            if pct_b_actual > 0.8:
                bb_orden = "BREAKOUT ALCISTA"
            elif pct_b_actual < 0.2:
                bb_orden = "BREAKOUT BAJISTA"
            else:
                bb_orden = "NO OPERAR"
        else:
            if pct_b_actual >= 1.0:
                bb_orden = "ENTRAR/MANTENER LARGO"
            elif pct_b_actual <= 0.0:
                bb_orden = "VENDER/CORTO MASIVO"
            elif 0.4 <= pct_b_actual <= 0.6:
                bb_orden = "LIQUIDAR O MANTENER CASH"
            elif pct_b_actual > 0.6:
                bb_orden = "TENDENCIA ALCISTA"
            elif pct_b_actual < 0.4:
                bb_orden = "TENDENCIA BAJISTA"

        # -- RSI --
        rsi_actual = float(df['RSI_14'].iloc[-1])
        rsi_estado, rsi_orden = "Zona Neutral", "ESPERAR"
        if rsi_actual < 30:
            rsi_estado, rsi_orden = "Sobreventa Extrema", "COMPRAR"
        elif 30 <= rsi_actual <= 45:
            rsi_estado, rsi_orden = "Desgaste Bajista", "MANTENER"
        elif 45 < rsi_actual <= 55:
            pass
        elif 55 < rsi_actual <= 70:
            rsi_estado, rsi_orden = "Tendencia Alcista", "MANTENER"
        else:
            rsi_estado, rsi_orden = "Sobrecompra Extrema", "VENDER"

        # -- MACD --
        macd_actual, macd_previo = float(df['MACD_Line'].iloc[-1]), float(df['MACD_Line'].iloc[-2])
        signal_actual, signal_previo = float(df['Signal_Line'].iloc[-1]), float(df['Signal_Line'].iloc[-2])
        hist_actual, hist_previo = float(df['Histograma'].iloc[-1]), float(df['Histograma'].iloc[-2])
        macd_estado, macd_orden = "Transición/Ruido", "ESPERAR"

        if (macd_previo < signal_previo) and (macd_actual > signal_actual) and (macd_actual < 0):
            macd_estado, macd_orden = "Rebote", "COMPRAR"
        elif (macd_actual > signal_actual) and (macd_actual > 0):
            if hist_actual > hist_previo:
                macd_estado, macd_orden = "T. Alcista Fuerte", "MANTENER"
            else:
                macd_estado, macd_orden = "D. Alcista", "TOMAR GANANCIAS"
        elif (macd_previo > signal_previo) and (macd_actual < signal_actual) and (macd_actual > 0):
            macd_estado, macd_orden = "Corrección", "VENDER"
        elif (macd_actual < signal_actual) and (macd_actual < 0):
            if hist_actual < hist_previo:
                macd_estado, macd_orden = "T. Bajista Fuerte", "ESPERAR"
            else:
                macd_estado, macd_orden = "Pérdida de Inercia Bajista", "OBSERVAR"

        # -- OBV --
        precio_sma_act = float(df['Precio_SMA20'].iloc[-1]) if 'Precio_SMA20' in df.columns else float(df['Close'].rolling(20).mean().iloc[-1])
        obv_actual = float(df['OBV'].iloc[-1])
        obv_sma_act = float(df['OBV_SMA20'].iloc[-1])
        
        tendencia_precio = cierre_actual > precio_sma_act
        tendencia_volumen = obv_actual > obv_sma_act
        obv_estado, obv_orden = "Transición Estocástica", "ESPERAR CONFIRMACIÓN"

        if tendencia_precio and tendencia_volumen:
            obv_estado, obv_orden = "Confluencia Alcista (Smart Money Buying)", "COMPRAR / MANTENER LARGO"
        elif not tendencia_precio and not tendencia_volumen:
            obv_estado, obv_orden = "Confluencia Bajista (Liquidación Masiva)", "VENDER / PERMANECER EN CASH"
        elif tendencia_precio and not tendencia_volumen:
            obv_estado, obv_orden = "Divergencia Bajista (Rally sin Volumen)", "ALERTA TRAMPA DE TOROS TOMAR GANANCIAS"
        elif not tendencia_precio and tendencia_volumen:
            obv_estado, obv_orden = "Divergencia Alcista (Acumulación Sigilosa)", "PREPARAR COMPRA (Esperar quiebre Precio)"

        # -- ADX / DMI --
        adx_val = float(df['ADX'].iloc[-1])
        adx_prev = float(df['ADX'].iloc[-2])
        di_plus = float(df['+DI'].iloc[-1])
        di_minus = float(df['-DI'].iloc[-1])
        adx_estado, adx_orden = "Lateral / Ruido", "LIQUIDAR / MANTENER CASH"

        if adx_val > 25:
            if di_plus > di_minus:
                if (adx_val < adx_prev) and (adx_val > 40):
                    adx_estado, adx_orden = "T. Alcista Exhausta", "GANANCIAS"
                else:
                    adx_estado, adx_orden = "T. Alcista", "E/S LARGO"
            elif di_minus > di_plus:
                if (adx_val < adx_prev) and (adx_val > 40):
                    adx_estado, adx_orden = "Pánico Vendedor Agotado", "CUBRIR CORTOS"
                else:
                    adx_estado, adx_orden = "T. Bajista", "E/S CORTO"
        elif adx_val <= 25:
            adx_estado, adx_orden = "C. Horizontal", "APAGAR"

        # -- Parabolic SAR --
        sar_actual = float(df['SAR'].iloc[-1])
        trend_actual = int(df['SAR_Trend'].iloc[-1])
        trend_previo = int(df['SAR_Trend'].iloc[-2])
        ema_actual = float(df['EMA_50'].iloc[-1])
        
        riesgo_stop_pct = abs((cierre_actual - sar_actual) / cierre_actual) * 100
        sar_estado, sar_orden = "Auditoria en Progreso", "ESPERAR"

        if cierre_actual > ema_actual:
            if trend_actual == 1 and trend_previo == -1:
                sar_estado, sar_orden = "Ruptura Parabólica", "COMPRAR"
            elif trend_actual == 1:
                if riesgo_stop_pct < 1.5:
                    sar_estado, sar_orden = "Trailing Stop Cerca", "MANTENER"
                else:
                    sar_estado, sar_orden = "Tendencia Sólida", "MANTENER POSICIÓN"
            elif trend_actual == -1 and trend_previo == 1:
                sar_estado, sar_orden = "Gatillo SAR Activado", "CERRAR POSICIÓN"
            elif trend_actual == -1:
                sar_estado, sar_orden = "EMA > Cierre > SAR", "NO COMPRAR"
        else:
            sar_estado, sar_orden = "Mercado Débil", "MANTENER EN LIQUIDEZ"

        # 5. Estructura de Salida JSON Estructurada
        return {
            "Activo": ticker,
            "Fecha": fecha_corte,
            "Cierre_USD": round(cierre_actual, 2),
            "Fase_Mercado_Macro": mm_estado,
            "Fuerza_ADX": round(adx_val, 2),
            "RSI_14": round(rsi_actual, 2),
            "SAR_Distancia_%": round(riesgo_stop_pct, 2),
            "Diagnostico_MACD": macd_estado,
            "Mandato_OBV": obv_orden,
            "Mandato_ADX": adx_orden,
            "Mandato_SAR": sar_orden,
            "Mandato_Global_Sugerido": "ALERTA: VOLATILIDAD" if es_squeeze else mm_orden
        }

    except Exception as e:
        return {"Activo": ticker, "Error": str(e)}

def ejecutar_escaneo():
    """
    Función principal que barre la cesta de activos, genera JSON y actualiza CSV.
    """
    # Cesta representativa
    cesta_activos = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "BTC-USD"]
    
    print("Inicializando Escáner de Mercado Institucional...")
    resultados = [auditar_activo(ticker) for ticker in cesta_activos]
    
    # Asegurar que el directorio existe
    os.makedirs('data', exist_ok=True)
    
    # 1. Guardar estado actual en JSON (Para Streamlit)
    with open('data/latest_market_scan.json', 'w') as f:
        json.dump(resultados, f, indent=4)
        
    # 2. Anexar al log histórico en CSV
    df_resultados = pd.DataFrame(resultados)
    csv_path = 'data/signals_log.csv'
    
    if os.path.exists(csv_path):
        df_resultados.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_resultados.to_csv(csv_path, index=False)
        
    print(f"Escaneo finalizado. {len(resultados)} activos procesados.")

if __name__ == "__main__":
    ejecutar_escaneo()