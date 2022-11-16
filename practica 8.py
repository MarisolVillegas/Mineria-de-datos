"""
Marisol Villegas Rincon 1898149
"""
import requests
import io
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numbers
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split

#Data Importing

df = pd.read_csv("autoss.csv", encoding = 'ISO-8859-1')

url = "https://raw.githubusercontent.com/MarisolVillegas/Mineria-de-datos/main/autos.csv"
print("\nDATAFRAME\n",df)

label = df.pop('price')
label=np.log(label)
data_train, data_test, label_train, label_test = train_test_split(df, label, test_size = 0.2, random_state = 500)

def label_transform(df: pd.DataFrame, columns: list):
    for c in columns:
        lbl = preprocessing.LabelEncoder()
        df[c] = lbl.fit_transform(df[c].astype(str))

columns=['name', 'yearOfRegistration', 'powerPS']

label_transform(data_train, columns)
label_transform(data_test, columns)



xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(data_train, label_train)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

dtest=xgb.DMatrix(data_test)
predicty = np.exp2(model.predict(dtest))
Aprice=np.exp2(label_test)
out = pd.DataFrame({'Actual_Price': Aprice, 'predict_Price': predicty,'Diff' :(Aprice-predicty)})

print(out[['Actual_Price','predict_Price','Diff']].head(10))

sns.set(color_codes=True)
sns.regplot(out['Actual_Price'],out['Diff'], line_kws={"color":"red","alpha":0.5,"lw":4}, marker="x")
plt.savefig("aprice_regresion.png")