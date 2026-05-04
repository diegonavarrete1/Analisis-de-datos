# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

archivos = ['da_mpf_sis_2019.csv','DA_MPF_SIS_2020.csv', 'DA_MPF_SIS_2021.csv']
lista_tablas = []

for nombre in archivos:
    try:
        df_temp = pd.read_csv(nombre, encoding='latin-1')

        df_temp.columns = df_temp.columns.str.strip().str.lower()
        lista_tablas.append(df_temp)
        print(f"Cargado: {nombre}")
    except Exception as e:
        print(f"Error al cargar {nombre}: {e}")

df_completo = pd.concat(lista_tablas, ignore_index=True)
cols_pfm = [c for c in df_completo.columns if c.startswith('pfm')]
for col in cols_pfm:
    df_completo[col] = pd.to_numeric(df_completo[col], errors='coerce').fillna(0)
df_completo['total_metodos'] = df_completo[cols_pfm].sum(axis=1)
nacional = df_completo.groupby(['anio', 'mes'])['total_metodos'].sum().reset_index()

nacional['fecha'] = pd.to_datetime(
    nacional['anio'].astype(int).astype(str) + '-' +
    nacional['mes'].astype(int).astype(str) + '-01'
)

print(f"Años detectados: {nacional['anio'].unique()}") # 'anio' entre comillas
print(f"Meses detectados: {len(nacional)}")
print("Cantidad de tablas cargadas:", len(lista_tablas))

import matplotlib.ticker as ticker

plt.figure(figsize=(12, 5))
ax = plt.gca()

plt.plot(nacional.index, nacional['total_metodos'], marker='o', color = 'blue', linewidth=2, label='Total Nacional')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.title('Serie de Tiempo: Métodos de Planificación Familiar Nacional (2019-2021)')
plt.xlabel('Tiempo')
plt.ylabel('Número Total de Métodos de Planificación')

plt.legend()

plt.grid(True, alpha = 0.3)

plt.show()

"Descomposición de la serie"

resultado = seasonal_decompose(nacional['total_metodos'], model='additive', period=12)

fig,(ax1,ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(12,16), sharex=True)

def formato_millones(x, pos):
    return '{:,.0f}'.format(x)

ax1.plot(resultado.observed, color='pink', linewidth=1.5)
ax1.set_title(' Datos Observados (Total pfm01-pfm11)', fontsize=12)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(formato_millones))

ax2.plot(resultado.trend, color='red', linewidth=2)
ax2.set_title('1. Tendencia', fontsize=12)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(formato_millones))

ax3.plot(resultado.seasonal, color='green', linewidth=1.5)
ax3.set_title('2. Estacionalidad', fontsize=12)
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(formato_millones))

ax4.scatter(resultado.resid.index, resultado.resid, color='gray', s=20)
ax4.set_title('3. Parte Aleatoria', fontsize=12)
ax4.yaxis.set_major_formatter(ticker.FuncFormatter(formato_millones))

plt.tight_layout(pad=3.0)
fig.suptitle('Descomposición Serie (2019-2021)', fontsize=16, y=1.02)
plt.show()


serie_completa = nacional['total_metodos']
serie_completa = serie_completa.fillna(0)
serie_completa = serie_completa.astype(float)



decomp = seasonal_decompose(serie_completa, model='additive', period=12)  # estacioanlidad anual
resid = decomp.resid.dropna()
plt.figure(figsize=(12,5))
plot_acf(resid, lags=len(resid)-1, title='Autocorrelación (ACF) - Residuos') # autocorrelación total
plt.show()

plt.figure(figsize=(12,5))
plot_pacf(resid, lags=(len(resid)-1)//2, method='ols', title='Autocorrelación Parcial (PACF) - Residuos') #

from statsmodels.tsa.stattools import adfuller


serie_transformada = nacional['total_metodos'].diff(1).diff(12).dropna()


resultado_adf = adfuller(serie_transformada)


p_value = resultado_adf[1]

print("--- PRUEBA DE ESTACIONARIEDAD (DICKEY-FULLER AUMENTADA) ---")
print(f"Estadístico ADF: {resultado_adf[0]:.4f}")
print(f"p-value: {p_value:.6f}")
print("-" * 50)

if p_value < 0.05:
    print("CONCLUSIÓN:")
    print("El p-value es menor a 0.05. Se rechaza la hipótesis nula.")
    print("¡La serie transformada ES ESTACIONARIA y está lista para modelar!")
else:
    print("CONCLUSIÓN:")
    print("El p-value es mayor a 0.05. No se rechaza la hipótesis nula.")
    print("La serie NO es estacionaria todavía.")


plt.figure(figsize=(12, 4))
plt.plot(nacional['fecha'].iloc[13:], serie_transformada, color='blue', marker='o', markersize=4)
plt.title('Serie Transformada (Estacionaria) - Lista para el modelo', fontsize=14)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Tiempo')
plt.ylabel('Diferencias de Planificación')
plt.grid(True, alpha=0.3)
plt.show()

!pip install pmdarima

import pmdarima as pm

print("Buscando el mejor modelo ARIMA...") #(m=12)
modelo_arima = pm.auto_arima(nacional['total_metodos'],
                             seasonal=True, m=12,
                             d=1, D=1, # Le decimos que aplique 1 diferencia normal y 1 estacional
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)

print("\n--- EL MEJOR MODELO ---")
print(modelo_arima.summary())


meses_a_predecir = 12 # Predicciones para el próximo año
predicciones, int_confianza = modelo_arima.predict(n_periods=meses_a_predecir, return_conf_int=True)

ultima_fecha = nacional['fecha'].iloc[-1] #(iniciando donde terminó 2021)
fechas_prediccion = pd.date_range(start=ultima_fecha + pd.DateOffset(months=1), periods=meses_a_predecir, freq='MS')

import matplotlib.ticker as ticker


plt.figure(figsize=(12, 5))


plt.plot(nacional['fecha'], nacional['total_metodos'], label='Demanda Real (2019-2021)', color='blue', marker='o', markersize=4)


fecha_puente = [nacional['fecha'].iloc[-1]] + list(fechas_prediccion)
valor_puente = [nacional['total_metodos'].iloc[-1]] + list(predicciones)


plt.plot(fecha_puente, valor_puente, label='Pronóstico 2022', color='green', linestyle='--', marker='o', markersize=4)


plt.fill_between(fechas_prediccion,
                 int_confianza[:, 0],
                 int_confianza[:, 1],
                 color='grey', alpha=0.2, label='Margen de Error (95%)')


plt.title('Pronóstico Demanda de Métodos de Planificación Familiar (2022)', fontsize=14)
plt.xlabel('Tiempo')
plt.ylabel('Total de Métodos')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

import numpy as np
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. Ajuste del modelo sobre LOGARITMOS (como en la imagen)
print("Buscando el mejor modelo ARIMA sobre datos logarítmicos...")
nacional['log_total'] = np.log(nacional['total_metodos'])

modelo_arima = pm.auto_arima(nacional['log_total'],
                             seasonal=True, m=12,
                             d=1, D=1,
                             trace=True,
                             suppress_warnings=True,
                             stepwise=True)

# 2. Generar Predicciones (h = 12 meses para 2022)
meses_a_predecir = 12
pred_log, int_conf_log = modelo_arima.predict(n_periods=meses_a_predecir, return_conf_int=True)

# --- TRADUCCIÓN DEL CÓDIGO DE LA IMAGEN A PYTHON ---
# prediction$x <- exp(prediction$x)       -> Datos históricos
# prediction$mean <- exp(prediction$mean) -> Media del pronóstico
# prediction$lower <- exp(prediction$lower) -> Límite inferior
# prediction$upper <- exp(prediction$upper) -> Límite superior

historicos_exp = np.exp(nacional['log_total'])
media_exp = np.exp(pred_log)
inferior_exp = np.exp(int_conf_log[:, 0])
superior_exp = np.exp(int_conf_log[:, 1])

# 3. Graficación con estilo "autoplot"
plt.figure(figsize=(12, 6))

# Graficar histórico (Efecto inverso aplicado)
plt.plot(nacional['fecha'], historicos_exp, label='Datos Históricos (exp)', color='#333333', linewidth=1)

# Preparar fechas para el pronóstico
ultima_fecha = nacional['fecha'].iloc[-1]
fechas_pred = pd.date_range(start=ultima_fecha + pd.DateOffset(months=1), periods=meses_a_predecir, freq='MS')

# Graficar Pronóstico (Media)
plt.plot(fechas_pred, media_exp, label='Pronóstico 2022', color='blue', linewidth=2)

# Graficar Intervalos de Confianza (Sombreado azul como en la imagen)
plt.fill_between(fechas_pred, inferior_exp, superior_exp, color='blue', alpha=0.2, label='Intervalo de Confianza (95%)')

# Formato final y Título con notación ARIMA
plt.title(r'Pronóstico desde $ARIMA(0,1,1)(0,1,1)_{12}$ (Escala Real)', fontsize=14)
plt.xlabel('Tiempo')
plt.ylabel('Total de Métodos')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.show()

# -*- coding: utf-8 -*-
"""PROYECTO_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/185QPw20o1x-71xgNX8XO4PtJXft7rxm-
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

archivos = ['da_mpf_sis_2019.csv','DA_MPF_SIS_2020.csv', 'DA_MPF_SIS_2021.csv']
lista_tablas = []

for nombre in archivos:
    try:
        df_temp = pd.read_csv(nombre, encoding='latin-1')

        df_temp.columns = df_temp.columns.str.strip().str.lower()
        lista_tablas.append(df_temp)
        print(f"Cargado: {nombre}")
    except Exception as e:
        print(f"Error al cargar {nombre}: {e}")

df_completo = pd.concat(lista_tablas, ignore_index=True)
cols_pfm = [c for c in df_completo.columns if c.startswith('pfm')]
for col in cols_pfm:
    df_completo[col] = pd.to_numeric(df_completo[col], errors='coerce').fillna(0)
df_completo['total_metodos'] = df_completo[cols_pfm].sum(axis=1)
nacional = df_completo.groupby(['anio', 'mes'])['total_metodos'].sum().reset_index()

nacional['fecha'] = pd.to_datetime(
    nacional['anio'].astype(int).astype(str) + '-' +
    nacional['mes'].astype(int).astype(str) + '-01'
)

print("\n--- TERMINADO ---")
print(f"Años detectados: {nacional['anio'].unique()}") # 'anio' entre comillas
print(f"Meses detectados: {len(nacional)}")
print("Cantidad de tablas cargadas:", len(lista_tablas))

import matplotlib.ticker as ticker

plt.figure(figsize=(12, 5))
ax = plt.gca()

plt.plot(nacional.index, nacional['total_metodos'], marker='o', color = 'blue', linewidth=2, label='Total Nacional')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.title('Serie de Tiempo: Métodos de Planificación Familiar Nacional (2019-2021)')
plt.xlabel('Tiempo')
plt.ylabel('Número Total de Métodos de Planificación')

plt.legend()

plt.grid(True, alpha = 0.3)

plt.show()

"""Descomposición de la serie"""

resultado = seasonal_decompose(nacional['total_metodos'], model='additive', period=12)

fig,(ax1,ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(12,16), sharex=True)

def formato_millones(x, pos):
    return '{:,.0f}'.format(x)

ax1.plot(resultado.observed, color='pink', linewidth=1.5)
ax1.set_title(' Datos Observados (Total pfm01-pfm11)', fontsize=12)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(formato_millones))

ax2.plot(resultado.trend, color='red', linewidth=2)
ax2.set_title('1. Tendencia', fontsize=12)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(formato_millones))

ax3.plot(resultado.seasonal, color='green', linewidth=1.5)
ax3.set_title('2. Estacionalidad', fontsize=12)
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(formato_millones))

ax4.scatter(resultado.resid.index, resultado.resid, color='gray', s=20)
ax4.set_title('3. Parte Aleatoria', fontsize=12)
ax4.yaxis.set_major_formatter(ticker.FuncFormatter(formato_millones))

plt.tight_layout(pad=3.0)
fig.suptitle('Descomposición Serie (2019-2021)', fontsize=16, y=1.02)
plt.show()

"""**Gráficos para ACF y PACF (parte aleatoria)**"""

serie_completa = nacional['total_metodos']
serie_completa = serie_completa.fillna(0)
serie_completa = serie_completa.astype(float)



decomp = seasonal_decompose(serie_completa, model='additive', period=12)  # estacioanlidad anual
resid = decomp.resid.dropna()  # se eliminan Null
plt.figure(figsize=(12,5))
plot_acf(resid, lags=len(resid)-1, title='Autocorrelación (ACF) - Residuos') # autocorrelación total
plt.show()

plt.figure(figsize=(12,5))
plot_pacf(resid, lags=(len(resid)-1)//2, method='ols', title='Autocorrelación Parcial (PACF) - Residuos') #autocorrelación parcial
plt.show()

from statsmodels.tsa.stattools import adfuller


serie_transformada = nacional['total_metodos'].diff(1).diff(12).dropna()


resultado_adf = adfuller(serie_transformada)


p_value = resultado_adf[1]

print("--- PRUEBA DE ESTACIONARIEDAD (DICKEY-FULLER AUMENTADA) ---")
print(f"Estadístico ADF: {resultado_adf[0]:.4f}")
print(f"p-value: {p_value:.6f}")
print("-" * 50)

if p_value < 0.05:
    print("CONCLUSIÓN:")
    print("El p-value es menor a 0.05. Se rechaza la hipótesis nula.")
    print("¡La serie transformada ES ESTACIONARIA y está lista para modelar!")
else:
    print("CONCLUSIÓN:")
    print("El p-value es mayor a 0.05. No se rechaza la hipótesis nula.")
    print("La serie NO es estacionaria todavía.")


plt.figure(figsize=(12, 4))
plt.plot(nacional['fecha'].iloc[13:], serie_transformada, color='blue', marker='o', markersize=4)
plt.title('Serie Transformada (Estacionaria) - Lista para el modelo', fontsize=14)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Tiempo')
plt.ylabel('Diferencias de Planificación')
plt.grid(True, alpha=0.3)
plt.show()

!pip install pmdarima

import pmdarima as pm

print("Buscando el mejor modelo ARIMA...") #(m=12)
modelo_arima = pm.auto_arima(nacional['total_metodos'],
                             seasonal=True, m=12,
                             d=1, D=1, # Le decimos que aplique 1 diferencia normal y 1 estacional
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)

print("\n--- EL MEJOR MODELO ---")
print(modelo_arima.summary())

# ========== MODIFIED PREDICTION PART ==========
# Updated to use forecast approach with h=30 (like good_model_AP %>% forecast::forecast(h = 30))
meses_a_predecir = 30  # Changed from 12 to 30 months (h = 30)
predicciones, int_confianza = modelo_arima.predict(n_periods=meses_a_predecir, return_conf_int=True)

# Generate dates for the forecast period
ultima_fecha = nacional['fecha'].iloc[-1]  # Last date of historical data
fechas_prediccion = pd.date_range(start=ultima_fecha + pd.DateOffset(months=1),
                                   periods=meses_a_predecir,
                                   freq='MS')

import matplotlib.ticker as ticker

# Create the forecast plot (similar to autoplot() in R)
plt.figure(figsize=(14, 7))

# Plot historical data
plt.plot(nacional['fecha'], nacional['total_metodos'],
         label='Datos Históricos', color='blue', marker='o', markersize=4, linewidth=2)

# Connect last historical point to first forecast point for continuity
fecha_puente = [nacional['fecha'].iloc[-1]] + list(fechas_prediccion)
valor_puente = [nacional['total_metodos'].iloc[-1]] + list(predicciones)

# Plot forecast line
plt.plot(fecha_puente, valor_puente,
         label='Pronóstico', color='green', linestyle='--', marker='o', markersize=4, linewidth=2)

# Add confidence intervals (95%)
plt.fill_between(fechas_prediccion,
                 int_confianza[:, 0],
                 int_confianza[:, 1],
                 color='green', alpha=0.2, label='Intervalo de Confianza (95%)')

# Customize the plot with general_theme-like styling
plt.title(r'Pronóstico desde $ARIMA(0,1,1)(0,1,1)_{12}$', fontsize=16)
plt.xlabel('Tiempo', fontsize=12)
plt.ylabel('Total de Métodos', fontsize=12)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Display the plot
plt.show()

# Optional: Print forecast summary
print("\n--- PRONÓSTICO PARA LOS PRÓXIMOS 30 MESES ---")
forecast_df = pd.DataFrame({
    'Fecha': fechas_prediccion,
    'Pronóstico': predicciones,
    'Límite Inferior (95%)': int_confianza[:, 0],
    'Límite Superior (95%)': int_confianza[:, 1]
})
print(forecast_df.head(10))
print(f"\nRango del pronóstico: {fechas_prediccion[0]} a {fechas_prediccion[-1]}")
print(f"Total de períodos pronosticados: {meses_a_predecir} meses")
