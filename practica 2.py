"""
Marisol Villegas Rinconn 1898149
"""
 #return pd.read_csv("autos.csv")
#import requests
#import io
#from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
#from tabulate import tabulate
#from typing import Tuple, List
#import re
#from datetime import datetime


#Data Importing

df = pd.read_csv("autoss.csv", encoding = 'ISO-8859-1')

url = "https://raw.githubusercontent.com/MarisolVillegas/Mineria-de-datos/main/autos.csv"
#df = get_csv_from_url(url)
print("\nDATAFRAME\n",df)

#Data Cleaning 

#Ordena la columna de forma descendente 
df.sort_values(by="vehicleType", ascending=False)

#Agrupa vehicleType con la media de cada columna del dataframe  
vehicleType = df.groupby(['vehicleType']).mean()
print("\nvehicleType\n",vehicleType)

#Suma valores de columnas e imprime el precio Final
suma = vehicleType.apply(np.sum, axis=0)
print("\nPRECIO FINAL:\n",suma[1:2])

#Convierte dataframe a un CSV limpio 
df.to_csv('autoss.csv')

#Data Statistics
print("\nMuestra de Datos Estadisticos\n")

#primeros 5 registros, uktimos 5 e informacion general del df
print(df.head())
print(df.tail()) 
print(df.info(), df.describe())

#Media, mediana y correlacion de la columna Price
print('Media general de los precios: $',df['price'].mean())
print('Mediana General de los precios:',df['price'].median())
print('Desviacion estandar General de los precios:',df['price'].std())


print("Correlacion es:\n",df.corr())

#Cantidad de valores en los registros
print("\nCantidad de valores\n",df.count())

#Devuelve el maximo y el minimo de los valores sobre el eje solicitado
print("\nMaximo:\n",df.max(),"\nMinimo:\n",df.min())