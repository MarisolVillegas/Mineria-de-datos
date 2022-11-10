# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:53:09 2022

@author: sol_r
"""

import requests
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
#Data Importing

df = pd.read_csv("autoss.csv", encoding = 'ISO-8859-1')
df = df.drop(df[df['Unnamed: 0']==129969].index)
url = "https://raw.githubusercontent.com/MarisolVillegas/Mineria-de-datos/main/autos.csv"
#df = get_csv_from_url(url)
print("\nDATAFRAME\n",df)

from sklearn.model_selection import train_test_split

X = df[['powerPS']]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #20%
x_train.sample(5)

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(normalize=True).fit(x_train, y_train)

y_pred = linear_model.predict(x_test)

import seaborn as sns

#create scatterplot with regression line
sns.regplot(x_test, y_pred, ci=None)
