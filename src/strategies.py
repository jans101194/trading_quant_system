import numpy as np
import pandas as pd

def generar_posicion(df, cond_compra=None, cond_venta=None, cond_neutral=None):
    """
    Motor central unificado para inyectar señales en la serie de tiempo.
    Inicializa en NaN para permitir que ffill() propague el último estado conocido.
    """
    senal = pd.Series(np.nan, index=df.index)
    
    if cond_compra is not None:
        senal.loc[cond_compra] = 1.0
    if cond_venta is not None:
        senal.loc[cond_venta] = -1.0
    if cond_neutral is not None:
        senal.loc[cond_neutral] = 0.0
        
    return senal.ffill().fillna(0.0)

# ==========================================
# Medias Móviles (4 Estrategias)
# ==========================================
def estrategias_medias_moviles(df, mm_r=20, mm_l=50):
    # E1: Precio vs SMA Lenta
    c_e1 = (df['Close'] > df[f'SMA_{mm_l}']) & (df['Close'].shift(1) <= df[f'SMA_{mm_l}'].shift(1))
    v_e1 = (df['Close'] < df[f'SMA_{mm_l}']) & (df['Close'].shift(1) >= df[f'SMA_{mm_l}'].shift(1))
    df['Pos_MM_E1_Precio_vs_SMA'] = generar_posicion(df, cond_compra=c_e1, cond_venta=v_e1)
    
    # E2: SMA Rápida vs SMA Lenta
    c_e2 = (df[f'SMA_{mm_r}'] > df[f'SMA_{mm_l}']) & (df[f'SMA_{mm_r}'].shift(1) <= df[f'SMA_{mm_l}'].shift(1))
    v_e2 = (df[f'SMA_{mm_r}'] < df[f'SMA_{mm_l}']) & (df[f'SMA_{mm_r}'].shift(1) >= df[f'SMA_{mm_l}'].shift(1))
    df['Pos_MM_E2_Cruce_SMA'] = generar_posicion(df, cond_compra=c_e2, cond_venta=v_e2)
    
    # E3: Precio vs EMA Lenta
    c_e3 = (df['Close'] > df[f'EMA_{mm_l}']) & (df['Close'].shift(1) <= df[f'EMA_{mm_l}'].shift(1))
    v_e3 = (df['Close'] < df[f'EMA_{mm_l}']) & (df['Close'].shift(1) >= df[f'EMA_{mm_l}'].shift(1))
    df['Pos_MM_E3_Precio_vs_EMA'] = generar_posicion(df, cond_compra=c_e3, cond_venta=v_e3)
    
    # E4: EMA Rápida vs EMA Lenta
    c_e4 = (df[f'EMA_{mm_r}'] > df[f'EMA_{mm_l}']) & (df[f'EMA_{mm_r}'].shift(1) <= df[f'EMA_{mm_l}'].shift(1))
    v_e4 = (df[f'EMA_{mm_r}'] < df[f'EMA_{mm_l}']) & (df[f'EMA_{mm_r}'].shift(1) >= df[f'EMA_{mm_l}'].shift(1))
    df['Pos_MM_E4_Cruce_EMA'] = generar_posicion(df, cond_compra=c_e4, cond_venta=v_e4)
    return df

# ==========================================
# Bandas de Bollinger (2 Estrategias)
# ==========================================
def estrategias_bollinger(df, periodos=20):
    ub, lb, mb = f'UB_{periodos}', f'LB_{periodos}', f'MB_{periodos}'
    
    # E1: Reversión a la media
    c_e1 = (df['Close'] > df[lb]) & (df['Close'].shift(1) <= df[lb].shift(1))
    v_e1 = (df['Close'] < df[ub]) & (df['Close'].shift(1) >= df[ub].shift(1))
    df['Pos_BB_E1_Reversion'] = generar_posicion(df, cond_compra=c_e1, cond_venta=v_e1)
    
    # E2: Riding the Bands (Breakout Long-Only, salida en Media Central)
    c_e2 = (df['Close'] > df[ub]) & (df['Close'].shift(1) <= df[ub].shift(1))
    n_e2 = (df['Close'] < df[mb]) & (df['Close'].shift(1) >= df[mb].shift(1))
    df['Pos_BB_E2_Breakout'] = generar_posicion(df, cond_compra=c_e2, cond_neutral=n_e2)
    return df

# ==========================================
# Oscilador Estocástico (3 Estrategias)
# ==========================================
def estrategias_estocastico(df):
    # E1: Cruces estáticos (20 y 80)
    c_e1 = (df['%K'] > 20) & (df['%K'].shift(1) <= 20)
    v_e1 = (df['%K'] < 80) & (df['%K'].shift(1) >= 80)
    df['Pos_STOCH_E1_Extremos'] = generar_posicion(df, cond_compra=c_e1, cond_venta=v_e1)
    
    # E2: Cruce direccional K vs D
    c_e2 = (df['%K'] > df['%D']) & (df['%K'].shift(1) <= df['%D'].shift(1))
    v_e2 = (df['%K'] < df['%D']) & (df['%K'].shift(1) >= df['%D'].shift(1))
    df['Pos_STOCH_E2_Cruce_Puro'] = generar_posicion(df, cond_compra=c_e2, cond_venta=v_e2)
    
    # E3: Reversión confirmada (Filtro institucional en zona extrema)
    c_e3 = c_e2 & (df['%K'].shift(1) < 20) & (df['%D'].shift(1) < 20)
    v_e3 = v_e2 & (df['%K'].shift(1) > 80) & (df['%D'].shift(1) > 80)
    df['Pos_STOCH_E3_Filtro_Institucional'] = generar_posicion(df, cond_compra=c_e3, cond_venta=v_e3)
    return df

# ==========================================
# On-Balance Volume (4 Estrategias)
# ==========================================
def estrategias_obv(df, sma_obv=20, ema_obv=10, sma2_obv=30):
    # E1: OBV vs SMA
    c_e1 = (df['OBV'] > df[f'OBV_SMA{sma_obv}'])
    n_e1 = (df['OBV'] <= df[f'OBV_SMA{sma_obv}'])
    df['Pos_OBV_E1_OBV_vs_SMA'] = generar_posicion(df, cond_compra=c_e1, cond_neutral=n_e1)
    
    # E2: OBV vs EMA Rápida
    c_e2 = (df['OBV'] > df[f'OBV_EMA{ema_obv}'])
    n_e2 = (df['OBV'] <= df[f'OBV_EMA{ema_obv}'])
    df['Pos_OBV_E2_OBV_vs_EMA'] = generar_posicion(df, cond_compra=c_e2, cond_neutral=n_e2)

    # E3: Cruce de Medias de OBV
    c_e3 = (df[f'OBV_EMA{ema_obv}'] > df[f'OBV_SMA{sma2_obv}'])
    n_e3 = (df[f'OBV_EMA{ema_obv}'] <= df[f'OBV_SMA{sma2_obv}'])
    df['Pos_OBV_E3_Cruce_Medias'] = generar_posicion(df, cond_compra=c_e3, cond_neutral=n_e3)
    
    # E4: Filtro Dual (Precio y Volumen)
    c_e4 = (df['Close'] > df['SMA_20']) & (df['OBV'] > df[f'OBV_SMA{sma_obv}'])
    n_e4 = ~c_e4
    df['Pos_OBV_E4_Filtro_Dual'] = generar_posicion(df, cond_compra=c_e4, cond_neutral=n_e4)
    return df

# ==========================================
# MACD (3 Estrategias)
# ==========================================
def estrategias_macd(df):
    # E1: Cruce de la Línea MACD vs Señal
    c_e1 = (df['MACD_Line'] > df['Signal_Line']) & (df['MACD_Line'].shift(1) <= df['Signal_Line'].shift(1))
    v_e1 = (df['MACD_Line'] < df['Signal_Line']) & (df['MACD_Line'].shift(1) >= df['Signal_Line'].shift(1))
    df['Pos_MACD_E1_Cruce_Clasico'] = generar_posicion(df, cond_compra=c_e1, cond_venta=v_e1)
    
    # E2: Cruce de la Línea Cero
    c_e2 = (df['MACD_Line'] > 0) & (df['MACD_Line'].shift(1) <= 0)
    v_e2 = (df['MACD_Line'] < 0) & (df['MACD_Line'].shift(1) >= 0)
    df['Pos_MACD_E2_Cruce_Cero'] = generar_posicion(df, cond_compra=c_e2, cond_venta=v_e2)
    
    # E3: Inflexión del Histograma
    c_e3 = (df['Histograma'] < 0) & (df['Histograma'] > df['Histograma'].shift(1)) & (df['Histograma'].shift(1) < df['Histograma'].shift(2))
    v_e3 = (df['Histograma'] > 0) & (df['Histograma'] < df['Histograma'].shift(1)) & (df['Histograma'].shift(1) >= df['Histograma'].shift(2))
    df['Pos_MACD_E3_Inflexion_Hist'] = generar_posicion(df, cond_compra=c_e3, cond_venta=v_e3)
    return df

# ==========================================
# ADX (2 Estrategias)
# ==========================================
def estrategias_adx(df, umbral=25):
    # E1: DMI Crudo
    c_e1 = (df['+DI'] > df['-DI']) & (df['+DI'].shift(1) <= df['-DI'].shift(1))
    v_e1 = (df['-DI'] > df['+DI']) & (df['-DI'].shift(1) <= df['+DI'].shift(1))
    df['Pos_ADX_E1_DMI_Crudo'] = generar_posicion(df, cond_compra=c_e1, cond_venta=v_e1)
    
    # E2: DMI Integral (+DI vs -DI filtrado por ADX)
    c_e2 = (df['+DI'] > df['-DI']) & (df['ADX'] > umbral)
    v_e2 = (df['-DI'] > df['+DI']) & (df['ADX'] > umbral)
    n_e2 = (df['ADX'] <= umbral)
    df['Pos_ADX_E2_Filtro_Integral'] = generar_posicion(df, cond_compra=c_e2, cond_venta=v_e2, cond_neutral=n_e2)
    return df

# ==========================================
# RSI (4 Estrategias)
# ==========================================
def estrategias_rsi(df, rsi_p=14, sma_rsi=9):
    # E1: Clásica 70/30 (Salida de Zona Extrema)
    c_e1 = (df[f'RSI_{rsi_p}'] > 30) & (df[f'RSI_{rsi_p}'].shift(1) <= 30)
    v_e1 = (df[f'RSI_{rsi_p}'] < 70) & (df[f'RSI_{rsi_p}'].shift(1) >= 70)
    df['Pos_RSI_E1_Clasica_70_30'] = generar_posicion(df, cond_compra=c_e1, cond_venta=v_e1)
    
    # E2: Línea Central (50)
    c_e2 = (df[f'RSI_{rsi_p}'] > 50) & (df[f'RSI_{rsi_p}'].shift(1) <= 50)
    v_e2 = (df[f'RSI_{rsi_p}'] < 50) & (df[f'RSI_{rsi_p}'].shift(1) >= 50)
    df['Pos_RSI_E2_Linea_Central'] = generar_posicion(df, cond_compra=c_e2, cond_venta=v_e2)
    
    # E3: Gatillo Dinámico (RSI vs SMA)
    c_e3 = (df[f'RSI_{rsi_p}'] > df[f'SMA{sma_rsi}_del_RSI']) & (df[f'RSI_{rsi_p}'].shift(1) <= df[f'SMA{sma_rsi}_del_RSI'].shift(1))
    v_e3 = (df[f'RSI_{rsi_p}'] < df[f'SMA{sma_rsi}_del_RSI']) & (df[f'RSI_{rsi_p}'].shift(1) >= df[f'SMA{sma_rsi}_del_RSI'].shift(1))
    df['Pos_RSI_E3_Gatillo_Dinamico'] = generar_posicion(df, cond_compra=c_e3, cond_venta=v_e3)
    
    # E4: Extrema 80/20
    c_e4 = (df[f'RSI_{rsi_p}'] > 20) & (df[f'RSI_{rsi_p}'].shift(1) <= 20)
    v_e4 = (df[f'RSI_{rsi_p}'] < 80) & (df[f'RSI_{rsi_p}'].shift(1) >= 80)
    df['Pos_RSI_E4_Extrema_80_20'] = generar_posicion(df, cond_compra=c_e4, cond_venta=v_e4)
    return df

# ==========================================
# Parabolic SAR (2 Estrategias)
# ==========================================
def estrategias_sar(df, ema_l=50):
    # E1: SAR Puro
    df['Pos_SAR_E1_Puro'] = df['SAR_Trend']
    
    # E2: SAR Filtrado Institucional (Long-Only)
    c_e2 = (df['SAR_Trend'] == 1) & (df['Close'] > df[f'EMA_{ema_l}'])
    n_e2 = (df['SAR_Trend'] == -1)
    df['Pos_SAR_E2_Filtro_EMA'] = generar_posicion(df, cond_compra=c_e2, cond_neutral=n_e2)
    return df