# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:06:40 2022

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
from sklearn.cluster import KMeans
#Data Importing

df = pd.read_csv("autoss.csv", encoding = 'ISO-8859-1')
df = df.drop(df[df['Unnamed: 0']==129969].index)
url = "https://raw.githubusercontent.com/MarisolVillegas/Mineria-de-datos/main/autos.csv"
#df = get_csv_from_url(url)
print("\nDATAFRAME\n",df)

def generate_df(means: List[Tuple[float, float, str]], n: int) -> pd.DataFrame:
    lists = [
        (df["powerPS"], df["price"], df["abtest"])
        for _x, _y, _l in means
    ]
    x = np.array([])
    y = np.array([])
    labels = np.array([])
    for _x, _y, _l in lists:
        x = np.concatenate((x, _x), axis=None)
        y = np.concatenate((y, _y))
        labels = np.concatenate((labels, _l))
    return pd.DataFrame({"x": x, "y": y, "label": labels})

def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

def scatter_group_by(
    file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots(figsize=(17,6))
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend()
    #plt.plot(figsize=(4,4))
    plt.savefig(file_path)
    plt.close()

#KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(df["x"], df["y"])

groups = [(1825, 10000, "grupo1"), (1900, 31000, "grupo2"), (1975, 3.3e+06, "grupo3")]
df = generate_df(groups, 100)
#filtro = df['x'] < 400000
#df = df[filtro]
dfC = pd.DataFrame(df, columns=['x', 'y'])

kmeans = KMeans(n_clusters=3).fit(dfC)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
plt.savefig("Clustering_groups.png")
plt.close()

