
import pandas as pd
import numpy as np
import yfinance as yf

# Descargar precios desde Yahoo Finance
tickers = ['IVV', 'ILF', 'SOXX']
vector_w = [0.5, 0.3, 0.2]
vector_w = np.array(vector_w).reshape(-1, 1)

df_prices = yf.download(
    tickers,
    start='2024-01-01',
    end='2026-09-30'
)['Close']

# Mantener el mismo orden de los pesos
df_prices = df_prices[tickers]

# Calcular retornos diarios
df_ret = df_prices.pct_change().dropna()

# Calcular matriz varianza covarianza
df_cov = df_ret.cov()
print(df_cov)

# Calcular varianza del portafolio
w = vector_w
w_t = w.T
sigma = df_cov.values

print('w', w.shape)
print('w_t', w_t.shape)
print('sigma', sigma.shape)

varianza = np.dot(sigma, w)
varianza = np.dot(w_t, varianza)[0][0]

desv = varianza ** (1/2)
volatilidad_anual = desv * (252 ** (1/2))

print(f'La volatilidad anual del portafolio es: {volatilidad_anual*100:.4f}%')

# Calcular la media del portafolio
vector_ret = df_ret.mean()
vector_ret = vector_ret.values
vector_ret = vector_ret.reshape(-1, 1)

mu = np.dot(w_t, vector_ret)[0][0]
mu_anual = mu * 252

print(f'La media anual del portafolio es: {mu_anual*100:.4f}%')

# Calcular Value at Risk (VaR) al 95% de confianza
var_95 = mu_anual - 1.64 * volatilidad_anual

print(f'El umbral anual de retorno al 95% es: {var_95*100:.4f}%')
