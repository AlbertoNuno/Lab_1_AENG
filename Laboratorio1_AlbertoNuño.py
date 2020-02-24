import Funciones as fn
import pandas as pd
from Datos import token
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as st
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from RegscorePy import aic
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
def volatility_forecast(instrument,granularity,days_to_forecast):
    significant_coef = lambda i : i if i>.5 else None
    rendimientos =returns(instrument,granularity)
    arima = ARIMA(rendimientos,order=(2,1,2))
    arima_fit = arima.fit(disp=0)
    residuals = pd.DataFrame(arima_fit.resid)
    residuals= residuals**2
    acf_coef = acf(residuals)
    q =list([significant_coef(i) for i in acf_coef])
    q =len(list(i for i in q if i))
    pacf_coef = pacf(residuals)
    p = list([significant_coef(i)for i in pacf_coef])
    p = len(list([i for i in p if i ]))
    garch_model = arch_model(residuals,mean="Zero",vol='GARCH',p=p,q=q)
    garch_model_fit=garch_model.fit()
    vol_forecast = garch_model_fit.forecast(horizon=days_to_forecast)
    return np.sqrt(vol_forecast.variance.values[-1:])

def next_direction(instrument,granularity,days_to_forecast):
    rendimientos = returns(instrument,granularity)
    vol=volatility_forecast(instrument,granularity,days_to_forecast)
    return rendimientos[-1] *vol


