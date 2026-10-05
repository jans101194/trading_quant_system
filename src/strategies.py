import numpy as np
import pandas as pd

def generar_posicion(df, condicion_compra, condicion_venta, neutralizar=False):
    senal = pd.Series(0.0, index=df.index)
    senal.loc[condicion_compra] = 1.0
    if neutralizar:
        senal.loc[condicion_venta] = 0.0
    else:
        senal.loc[condicion_venta] = -1.0
    posicion = senal.replace(0.0, np.nan).ffill().fillna(0.0)
    return posicion

def estrategias_medias_moviles(df):
    df['Estado_E1_MM'] = np.where(df['Close'] > df['SMA_50'], 1.0, 0.0)
    df['Estado_E2_MM'] = np.where(df['SMA_20'] > df['SMA_50'], 1.0, 0.0)
    df['Estado_E3_MM'] = np.where(df['Close'] > df['EMA_50'], 1.0, 0.0)
    df['Estado_E4_MM'] = np.where(df['EMA_20'] > df['EMA_50'], 1.0, 0.0)
    df['Senal_E1_MM'] = df['Estado_E1_MM'].diff()
    df['Senal_E2_MM'] = df['Estado_E2_MM'].diff()
    df['Senal_E3_MM'] = df['Estado_E3_MM'].diff()
    df['Senal_E4_MM'] = df['Estado_E4_MM'].diff()
    return df

def estrategias_bollinger(df):
    compra_e1 = (df['Close'] < df['LB']) & (df['Close'].shift(1) > df['LB'].shift(1))
    venta_e1 = (df['Close'] > df['UB']) & (df['Close'].shift(1) <= df['UB'].shift(1))
    df['Pos_E1_BB'] = generar_posicion(df, compra_e1, venta_e1, neutralizar=False)
    
    compra_e2 = (df['Close'] > df['UB']) & (df['Close'].shift(1) <= df['UB'].shift(1))
    venta_e2 = (df['Close'] < df['MB']) & (df['Close'].shift(1) > df['MB'].shift(1))
    df['Pos_E2_BB'] = generar_posicion(df, compra_e2, venta_e2, neutralizar=True)
    return df

def estrategias_estocastico(df):
    compra_e1 = (df['%K'] > 20) & (df['%K'].shift(1) <= 20)
    venta_e1 = (df['%K'] < 80) & (df['%K'].shift(1) >= 80)
    df['Pos_E1_STOCH'] = generar_posicion(df, compra_e1, venta_e1)
    
    compra_e2 = (df['%K'] > df['%D']) & (df['%K'].shift(1) < df['%D'].shift(1))
    venta_e2 = (df['%K'] < df['%D']) & (df['%K'].shift(1) > df['%D'].shift(1))
    df['Pos_E2_STOCH'] = generar_posicion(df, compra_e2, venta_e2)
    
    compra_e3 = compra_e2 & (df['%K'].shift(1) < 20) & (df['%D'].shift(1) < 20)
    venta_e3 = venta_e2 & (df['%K'].shift(1) > 80) & (df['%D'].shift(1) > 80)
    df['Pos_E3_STOCH'] = generar_posicion(df, compra_e3, venta_e3)
    return df

def estrategias_obv(df):
    df['Precio_SMA20'] = df['Close'].rolling(window=20).mean()
    df['Estado_E1_OBV'] = np.where(df['OBV'] > df['OBV_SMA20'], 1.0, 0.0)
    df['Estado_E2_OBV'] = np.where(df['OBV'] > df['OBV_EMA10'], 1.0, 0.0)
    df['Estado_E3_OBV'] = np.where(df['OBV_EMA10'] > df['OBV_SMA30'], 1.0, 0.0)
    
    condicion_precio = df['Close'] > df['Precio_SMA20']
    condicion_volumen = df['OBV'] > df['OBV_SMA20']
    df['Estado_E4_OBV'] = np.where(condicion_precio & condicion_volumen, 1.0, 0.0)
    
    df['Senal_E1_OBV'] = df['Estado_E1_OBV'].diff()
    df['Senal_E2_OBV'] = df['Estado_E2_OBV'].diff()
    df['Senal_E3_OBV'] = df['Estado_E3_OBV'].diff()
    df['Senal_E4_OBV'] = df['Estado_E4_OBV'].diff()
    return df

def estrategias_macd(df):
    compra_e1 = (df['MACD_Line'] > df['Signal_Line']) & (df['MACD_Line'].shift(1) <= df['Signal_Line'].shift(1))
    venta_e1 = (df['MACD_Line'] < df['Signal_Line']) & (df['MACD_Line'].shift(1) > df['Signal_Line'].shift(1))
    df['Pos_E1_MACD'] = generar_posicion(df, compra_e1, venta_e1)
    
    compra_e2 = (df['MACD_Line'] > 0) & (df['MACD_Line'].shift(1) <= 0)
    venta_e2 = (df['MACD_Line'] < 0) & (df['MACD_Line'].shift(1) >= 0)
    df['Pos_E2_MACD'] = generar_posicion(df, compra_e2, venta_e2)
    
    compra_e3 = (df['Histograma'] < 0) & (df['Histograma'] > df['Histograma'].shift(1)) & (df['Histograma'].shift(1) < df['Histograma'].shift(2))
    venta_e3 = (df['Histograma'] > 0) & (df['Histograma'] < df['Histograma'].shift(1)) & (df['Histograma'].shift(1) >= df['Histograma'].shift(2))
    df['Pos_E3_MACD'] = generar_posicion(df, compra_e3, venta_e3)
    return df

def estrategias_adx(df, umbral=25):
    cond_compra_e1 = (df['+DI'] > df['-DI']) & (df['+DI'].shift(1) < df['-DI'].shift(1))
    cond_venta_e1 = (df['-DI'] > df['+DI']) & (df['-DI'].shift(1) < df['+DI'].shift(1))
    df['Pos_E1_ADX'] = generar_posicion(df, cond_compra_e1, cond_venta_e1)
    
    cond_compra_e2 = (df['+DI'] > df['-DI']) & (df['ADX'] > umbral)
    cond_venta_e2 = (df['-DI'] > df['+DI']) & (df['ADX'] > umbral)
    
    senal_e2 = pd.Series(np.nan, index=df.index)
    senal_e2.loc[cond_compra_e2] = 1.0
    senal_e2.loc[cond_venta_e2] = -1.0
    senal_e2.loc[(df['ADX'] <= umbral)] = 0.0
    df['Pos_E2_ADX'] = senal_e2.ffill().fillna(0.0)
    return df

def estrategias_rsi(df):
    compra_e1 = (df['RSI_14'] > 30) & (df['RSI_14'].shift(1) <= 30)
    venta_e1 = (df['RSI_14'] < 70) & (df['RSI_14'].shift(1) >= 70)
    df['Pos_E1_RSI'] = generar_posicion(df, compra_e1, venta_e1)
    
    compra_e2 = (df['RSI_14'] > 50) & (df['RSI_14'].shift(1) <= 50)
    venta_e2 = (df['RSI_14'] < 50) & (df['RSI_14'].shift(1) >= 50)
    df['Pos_E2_RSI'] = generar_posicion(df, compra_e2, venta_e2)
    
    compra_e3 = (df['RSI_14'] > df['SMA9_del_RSI']) & (df['RSI_14'].shift(1) <= df['SMA9_del_RSI'].shift(1))
    venta_e3 = (df['RSI_14'] < df['SMA9_del_RSI']) & (df['RSI_14'].shift(1) >= df['SMA9_del_RSI'].shift(1))
    df['Pos_E3_RSI'] = generar_posicion(df, compra_e3, venta_e3)
    
    compra_e4 = (df['RSI_14'] > 20) & (df['RSI_14'].shift(1) <= 20)
    venta_e4 = (df['RSI_14'] < 80) & (df['RSI_14'].shift(1) >= 80)
    df['Pos_E4_RSI'] = generar_posicion(df, compra_e4, venta_e4)
    return df

def estrategias_sar(df):
    df['Pos_E1_SAR'] = df['SAR_Trend']
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    cond_compra_e2 = (df['SAR_Trend'] == 1) & (df['Close'] > df['EMA_50'])
    df['Pos_E2_SAR'] = np.where(cond_compra_e2, 1.0, 0.0)
    return df