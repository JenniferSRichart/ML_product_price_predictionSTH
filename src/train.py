from utils.funciones import mapping_REGION, mapping_SEGMENTO, mapping_ITEM, mapping_PROMO
import sys, os
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import pandas_profiling
import phik
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn import preprocessing, metrics, linear_model, model_selection
from sklearn.tree import DecisionTreeRegressor, plot_tree
from scipy.stats import shapiro, skew, iqr
from scipy import stats
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest
import statsmodels as sm
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

print("Comienza el el script train.py")

def get_root_path(n):
    '''
    Esta función nos permite iterar sobre carpetas para añadir el path de nuestra carpeta raíz
    Argumentos:
        - n (int): el número de veces que iteraremos para llegar a la carpeta deseada
    '''
    path = os.path.dirname(os.path.abspath(__file__)) #__file__ --> para .py
    for i in range(n):
        path = os.path.dirname(path)
    print(path)
    sys.path.append(path)

get_root_path(n=1)
#print(sys.path)

#print(str(os.getcwd()))

data = pd.read_excel('data/processed/Data_STH.xlsx')

# Creamos categorias Numericas
data['REGION_ID'] = data['REGION'].apply(mapping_REGION)
data['SEGMENTO_ID'] = data['AMBITO'].apply(mapping_SEGMENTO)
data['ITEM_ID'] = data['DESCR.ART.'].apply(mapping_ITEM)
data['PROMO_ID'] = data['PROMO'].apply(mapping_PROMO)
data['MONTH'] = data['Ciclo_MONTH'].apply(mapping_PROMO)
data['DATE'] = data['YEAR'] + data['MONTH'] + data['DAY']
data['DATE'] = pd.to_datetime(data['DATE'])

# Seleccionamos las variables necesarias
df = data[['REGION_ID','CICLO', 'YEAR', 'MONTH', 'ID_STANHOME', '%DTO.', 'PAGINA',
       'COD-MOV.', 'PVP','PVP_TOT_FACTURA', 'PROMO_ID']]

# Eliminamos los vacio
df.dropna()

# Creamos un Dataset por producto
df_1 = df[df["ITEM_ID"] == 1]
df_2 = df[df["ITEM_ID"] == 2]


# Guardamos
df_1.to_excel('df_1.xlsx', sheet_name='df_1')
df_2.to_excel('df_2.xlsx', sheet_name='df_2')

#Realizamos el Split
X1 = df_1.drop(['PVP'], axis=1)     
y1 = df_1['PVP']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.20, random_state = 42)

#Traemos el modelo aprendido
with open('../model/dtr.pkl','rb') as f:
    model_dtr = pickle.load(f)

predictions_dtr1 = model_dtr.predict(X1_test)
print('MAE_test:', metrics.mean_absolute_error(y1_test, predictions_dtr1))
print('MSE_test:', metrics.mean_squared_error(y1_test,predictions_dtr1))
print('RMSE_test:', np.sqrt(metrics.mean_squared_error(y1_test, predictions_dtr1)))
print('r2_SCORE_test:', r2_score(y1_test, predictions_dtr1))
sns.scatterplot(y1_test, predictions_dtr1);

#Probamos un precio nuevo
new_price1= [[1.0, 202115.0, 2021.0, 4.0, 24954.0, 18.0, 10.0, 2200.0, 71.0, 4.0]]
model_dtr.predict(new_price1)

print('Finished train.py')

