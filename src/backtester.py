import numpy as np
import pandas as pd

def calcular_rendimientos(df, col_precio='Close', columnas_posicion=None):
    if columnas_posicion is None:
        columnas_posicion = []
        
    df['Retorno_Mercado'] = df[col_precio].pct_change()
    columnas_retornos = ['Retorno_Mercado']
    
    for col_pos in columnas_posicion:
        nombre_ret = col_pos.replace('Pos_', 'Ret_')
        # Aplicamos shift(1) obligatorio en trading para evitar Look-Ahead Bias
        df[nombre_ret] = df[col_pos].shift(1) * df['Retorno_Mercado']
        columnas_retornos.append(nombre_ret)
        
    for col in columnas_retornos:
        df[f'Cum_{col}'] = (1 + df[col].fillna(0)).cumprod()
        
    return df, columnas_retornos

def calcular_metricas(df, columnas_retornos, nombres_estrategias):
    resultados = []
    nombres_completos = ['Buy & Hold (Mercado)'] + nombres_estrategias
    
    for nombre, col in zip(nombres_completos, columnas_retornos):
        retorno_total = (df[f'Cum_{col}'].iloc[-1] - 1) * 100
        std_dev = df[col].std()
        volatilidad_anual = std_dev * np.sqrt(252) * 100
        sharpe = (df[col].mean() / std_dev) * np.sqrt(252) if std_dev > 0 else 0
        
        # Max Drawdown (Caída máxima de capital)
        cum_ret = df[f'Cum_{col}']
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        resultados.append({
            'Estrategia': nombre,
            'Retorno Total (%)': round(retorno_total, 2),
            'Volatilidad Anual (%)': round(volatilidad_anual, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Sharpe Ratio': round(sharpe, 2)
        })
        
    return pd.DataFrame(resultados)