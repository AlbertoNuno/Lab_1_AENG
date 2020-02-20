import Funciones as fn
import pandas as pd
from Datos import token
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as st
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import pacf, acf
from arch import arch_model
from scipy.optimize import least_squares

def get_closes(instrument,granularity):
    f_inicio = pd.to_datetime("2015-01-06 00:00:00").tz_localize('GMT')
    if granularity == 'D':
        f_fin = pd.to_datetime((datetime.today()-timedelta(days=1)).strftime("%m/%d/%Y, %H:%M:%S")).tz_localize('GMT')
    else :
        f_fin = pd.to_datetime((datetime.now()).strftime("%m/%d/%Y, %H:%M:%S")).tz_localize('GMT')
    precios = fn.f_precios_masivos(p0_fini=f_inicio, p1_ffin=f_fin, p2_gran=granularity,
                                   p3_inst=instrument, p4_oatk=token, p5_ginc=4900)

    precios = pd.DataFrame(data={'High': precios.High, 'Low': precios.Low, 'Close': precios.Close
                                 }).astype('float')


    return precios.Close

def returns(instrument,granularity):
    precios = get_closes(instrument,granularity)
    rendimientos = (precios - precios.shift(1) / precios.shift(1)).iloc[1:]
    return rendimientos


def parametric_var(instrument,granularity,exposure,possition): ##devuelve el VaR a un día tomando un año de datos
    """
        Parameters
        ----------
        instrumet
        granularity
        exposure
        posstion
        Returns
        -------
        var
        ---------
        """
    rendimientos=returns(instrument,granularity)
    precios = get_closes(instrument,granularity)

    if possition == 'long':
        volatilidad = np.std(rendimientos)
        percentil = np.percentile(rendimientos, 1)
        valor_z = st.norm.ppf(percentil)
        var = valor_z * volatilidad * exposure*precios.iloc[-1]
    elif possition=='short':

        volatilidad = np.std(rendimientos)
        percentil = np.percentile(rendimientos, 99)
        valor_z = st.norm.ppf(percentil)
        var = valor_z * volatilidad * exposure*precios.iloc[-1]
    return var,precios.iloc[-1]

def estimate_autocorrelation(series):
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return round(acf_lag, 3)
    x = np.arange(n) # Avoiding lag 0 calculation
    acf_coeffs = list(map(r, x))
    return acf_coeffs

def modified_blackScholes(instrument,dt):
    """""
    Parameters
    ----------
    instrument
    dt

    """
    mu = np.mean(instrument)
    b= (pd.DataFrame(instrument.iloc[1:],instrument.shift(1)).cov())/instrument.shift(1)
    k = -np.log(b)/dt

    st= instrument.iloc[-1]
    s0=instrument.iloc[0]
    sigma = np.std(instrument.iloc[len(instrument)-250:])
    k=0
    st = (mu +.5*(sigma**2) -k*(np.log(st)-np.log(s0)-mu*dt) + (sigma*np.random.normal(0, dt)))

    return st





"""
rendimientos2=[i**2 for i in rendimientos]
acf = estimate_autocorrelation(rendimientos2)
pacf = pacf(rendimientos2,20)
model = arch_model(rendimientos,mean='Zero',vol='Garch',p=2,q=2)
model_fit = model.fit()
yhat = model_fit.forecast(horizon=1)
#fig.suptitle('ACF,PACF')
#plot_pacf(rendimientos2,lags=20)
#plot_pacf(rendimientos,lags=20)
"""
rendimientos = returns('USD_CAD','D')

arima_array=list()
for p in range(1,4):
    for q in range(1,4):
        arima_array=arima_array.append(ARIMA(rendimientos,(p,1,q)))




precios =get_closes("USD_CAD",'D')
var= parametric_var("USD_CAD",'D',800,'long')
precio_siguiente = modified_blackScholes(precios,1)
















