import numpy as np
import pandas as pd

# ==========================================
# Medias Móviles (SMA y EMA)
# ==========================================
def calcular_medias_moviles(df_mercado, col_precio='Close', ventana_rapida=20, ventana_lenta=50):
    df_mercado[f'SMA_{ventana_rapida}'] = df_mercado[col_precio].rolling(window=ventana_rapida).mean()
    df_mercado[f'SMA_{ventana_lenta}'] = df_mercado[col_precio].rolling(window=ventana_lenta).mean()
    df_mercado[f'EMA_{ventana_rapida}'] = df_mercado[col_precio].ewm(span=ventana_rapida, adjust=False).mean()
    df_mercado[f'EMA_{ventana_lenta}'] = df_mercado[col_precio].ewm(span=ventana_lenta, adjust=False).mean()
    return df_mercado

# ==========================================
# Bandas de Bollinger
# ==========================================
def calcular_bollinger(df, col_precio='Close', periodos=20, factor_k=2.0):
    df[f'MB_{periodos}'] = df[col_precio].rolling(window=periodos).mean()
    df[f'STD_{periodos}'] = df[col_precio].rolling(window=periodos).std(ddof=0)
    df[f'UB_{periodos}'] = df[f'MB_{periodos}'] + (df[f'STD_{periodos}'] * factor_k)
    df[f'LB_{periodos}'] = df[f'MB_{periodos}'] - (df[f'STD_{periodos}'] * factor_k)
    df['Bandwidth'] = (df[f'UB_{periodos}'] - df[f'LB_{periodos}']) / df[f'MB_{periodos}']
    df['Pct_B'] = (df[col_precio] - df[f'LB_{periodos}']) / (df[f'UB_{periodos}'] - df[f'LB_{periodos}'])
    return df

# ==========================================
# Oscilador Estocástico
# ==========================================
def calcular_estocastico(df_mercado, col_high='High', col_low='Low', col_close='Close', periodos_k=14, periodos_d=3):
    df_mercado[f'L{periodos_k}'] = df_mercado[col_low].rolling(window=periodos_k).min()
    df_mercado[f'H{periodos_k}'] = df_mercado[col_high].rolling(window=periodos_k).max()
    df_mercado['%K'] = 100 * ((df_mercado[col_close] - df_mercado[f'L{periodos_k}']) / (df_mercado[f'H{periodos_k}'] - df_mercado[f'L{periodos_k}']))
    df_mercado['%D'] = df_mercado['%K'].rolling(window=periodos_d).mean()
    return df_mercado

# ==========================================
# On-Balance Volume (OBV)
# ==========================================
def calcular_obv(df_mercado, col_precio='Close', col_volumen='Volume', ventana_sma=20, ventana_ema=10, ventana_sma2=30):
    diferencia_precio = df_mercado[col_precio].diff()
    direccion = np.sign(diferencia_precio).fillna(0)
    flujo_volumen = direccion * df_mercado[col_volumen]
    df_mercado['OBV'] = flujo_volumen.cumsum()
    df_mercado[f'OBV_SMA{ventana_sma}'] = df_mercado['OBV'].rolling(window=ventana_sma).mean()
    df_mercado[f'OBV_EMA{ventana_ema}'] = df_mercado['OBV'].ewm(span=ventana_ema, adjust=False).mean()
    df_mercado[f'OBV_SMA{ventana_sma2}'] = df_mercado['OBV'].rolling(window=ventana_sma2).mean()
    return df_mercado

# ==========================================
# MACD
# ==========================================
def calcular_macd(df_macd, col_precio='Close', n_rapida=12, n_lenta=26, n_senal=9):
    df_macd[f'MACD_EMA_{n_rapida}'] = df_macd[col_precio].ewm(span=n_rapida, adjust=False).mean()
    df_macd[f'MACD_EMA_{n_lenta}'] = df_macd[col_precio].ewm(span=n_lenta, adjust=False).mean()
    df_macd['MACD_Line'] = df_macd[f'MACD_EMA_{n_rapida}'] - df_macd[f'MACD_EMA_{n_lenta}']
    df_macd['Signal_Line'] = df_macd['MACD_Line'].ewm(span=n_senal, adjust=False).mean()
    df_macd['Histograma'] = df_macd['MACD_Line'] - df_macd['Signal_Line']
    return df_macd

# ==========================================
# Average Directional Index (ADX)
# ==========================================
def calcular_adx(df_mercado, col_high='High', col_low='Low', col_close='Close', n=14):
    prev_close = df_mercado[col_close].shift(1)
    tr1 = (df_mercado[col_high] - df_mercado[col_low]).abs()
    tr2 = (df_mercado[col_high] - prev_close).abs()
    tr3 = (df_mercado[col_low] - prev_close).abs()
    df_mercado['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = df_mercado[col_high] - df_mercado[col_high].shift(1)
    down_move = df_mercado[col_low].shift(1) - df_mercado[col_low]
    
    df_mercado['+DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    df_mercado['-DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    alpha_w = 1/n
    df_mercado['TR_suavizado'] = df_mercado['TR'].ewm(alpha=alpha_w, adjust=False).mean()
    df_mercado['+DM_suavizado'] = df_mercado['+DM'].ewm(alpha=alpha_w, adjust=False).mean()
    df_mercado['-DM_suavizado'] = df_mercado['-DM'].ewm(alpha=alpha_w, adjust=False).mean()
    
    df_mercado['+DI'] = 100 * (df_mercado['+DM_suavizado'] / df_mercado['TR_suavizado'])
    df_mercado['-DI'] = 100 * (df_mercado['-DM_suavizado'] / df_mercado['TR_suavizado'])
    
    df_mercado['DX'] = 100 * (abs(df_mercado['+DI'] - df_mercado['-DI']) / (df_mercado['+DI'] + df_mercado['-DI']))
    df_mercado['ADX'] = df_mercado['DX'].ewm(alpha=alpha_w, adjust=False).mean()
    return df_mercado

# ==========================================
# RSI (Relative Strength Index)
# ==========================================
def calcular_rsi(df_mercado, col_precio='Close', periodos_rsi=14, periodos_sma_rsi=9):
    delta = df_mercado[col_precio].diff()
    up = delta.clip(lower=0)
    down1 = -1 * delta.clip(upper=0)
    ema_up = up.ewm(alpha=1/periodos_rsi, adjust=False).mean()
    ema_down = down1.ewm(alpha=1/periodos_rsi, adjust=False).mean()
    rs = ema_up / ema_down
    df_mercado[f'RSI_{periodos_rsi}'] = 100 - (100 / (1 + rs))
    df_mercado[f'SMA{periodos_sma_rsi}_del_RSI'] = df_mercado[f'RSI_{periodos_rsi}'].rolling(window=periodos_sma_rsi).mean()
    return df_mercado

# ==========================================
# Parabolic SAR
# ==========================================
def calcular_parabolic_sar(df, col_high='High', col_low='Low', paso=0.02, max_af=0.20):
    n = len(df)
    sar = np.zeros(n)
    ep = np.zeros(n)
    af = np.zeros(n)
    trend = np.ones(n)
    
    sar[0] = df[col_low].iloc[0]
    ep[0] = df[col_high].iloc[0]
    af[0] = paso
    
    for i in range(1, n):
        sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
        if trend[i-1] == 1:
            if i >= 2:
                sar[i] = min(sar[i], df[col_low].iloc[i-1], df[col_low].iloc[i-2])
            if df[col_low].iloc[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = df[col_low].iloc[i]
                af[i] = paso
            else:
                trend[i] = 1
                if df[col_high].iloc[i] > ep[i-1]:
                    ep[i] = df[col_high].iloc[i]
                    af[i] = min(max_af, af[i-1] + paso)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:
            if i >= 2:
                sar[i] = max(sar[i], df[col_high].iloc[i-1], df[col_high].iloc[i-2])
            if df[col_high].iloc[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = df[col_high].iloc[i]
                af[i] = paso
            else:
                trend[i] = -1
                if df[col_low].iloc[i] < ep[i-1]:
                    ep[i] = df[col_low].iloc[i]
                    af[i] = min(max_af, af[i-1] + paso)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                    
    df['SAR'] = sar
    df['SAR_Trend'] = trend
    return df