
import pandas as pd
import numpy as np
import seaborn as sns
import sys, os
import matplotlib.pyplot as plt
from scipy.stats import shapiro, skew, iqr

def mapping_SEGMENTO(x):
    ''' Codifica los Segmento del Dataset en valor numerico'''
    if x == 'CO':
        return 1
    else:
        return 0

def mapping_REGION(x):
    ''' Codifica las Regiones del Dataset en valor numerico'''
    if x == 'CA':
        return 1
    elif x == 'GG':
        return 2
    elif x == 'SG':
        return 3
    elif x == 'MN':
        return 4
    else:
        return 5

def mapping_ITEM(x):
    ''' Codifica los Items del Dataset en valor numerico'''
    if x == 'La Crème Essentielle':
        return 1
    elif (x == 'All Purpose P.9104') or (x == 'ALL PURPOSE 300ML'):
        return 2
    elif x == 'CREMA REAFIRM COLLAGEN':
        return 3
    else:
        return 4

def mapping_PROMO(x):
    ''' Codifica las Promociones del Dataset en valor numerico'''
    if x == '2x1':
        return 1
    elif x == 'desc directo':
        return 2
    elif x == 'set':
        return 3
    elif x == 'set solo para altas':
        return 3
    else:
        return 4

def mapping_MONTH(x):
    ''' Codifica los Meses segun ciclo del Dataset en valor numerico'''
    if x == '01':
        return 1
    elif (x == '02') or (x == '03'):
        return 2
    elif x == '04':
        return 3
    elif (x == '05') or (x == '06'):
        return 4
    elif x == '07':
        return 5
    elif (x == '08') or (x == '09'):
        return 6
    elif x == '10':
        return 7
    elif x == '11':
        return 8
    elif (x == '12') or (x == '13'):
        return 9
    elif (x == '14') or (x == '15'):
        return 10
    elif x == '16':
        return 11
    else:
        return 12


def data_report(df):
    ''' Indica metricas como tipo de dato, valores unicos y  valores faltantes'''
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)
    
    return concatenado.T


def plot_df(df, x, y, title, xlabel, ylabel, dpi=100):
    ''' Grafica figura en base a 2 variables'''
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

def outliers_quantie(df, feature, param=1.5):     
    iqr_ = iqr(df[feature], nan_policy='omit')
    q1 = np.nanpercentile(df[feature], 25)
    q3 = np.nanpercentile(df[feature], 75)
    
    th1 = q1 - iqr_*param
    th2 = q3 + iqr_*param
    
    return df[(df[feature] >= th1) & (df[feature] <= th2)].reset_index(drop=True)

def get_root_path(n):
    '''
    Esta función nos permite iterar sobre carpetas para añadir el path de nuestra carpeta raíz
    Argumentos:
        - n (int): el número de veces que iteraremos para llegar a la carpeta deseada
    '''
    path = os.getcwd() # para notebook ||| __file__ --> para .py
    for i in range(n):
        path = os.path.dirname(path)
    print(path)
    sys.path.append(path)