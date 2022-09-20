# -*- coding: utf-8 -*-
"""
Marisol Villegas Rincón 1898149
"""
import requests
import io
#from bs4 import BeautifulSoup
import pandas as pd
#from tabulate import tabulate
#from typing import Tuple, List
#import re
#from datetime import datetime
#import numpy as np


def get_csv_from_url(url:str) -> pd.DataFrame:
        s = requests.get(url).content
        print("Lectura exitosa")
        return pd.read_csv(io.StringIO(s.decode())) 
        print("\nSe leyó de un archivo local")
        
       

url = "https://raw.githubusercontent.com/MarisolVillegas/Mineria-de-datos/main/autos.csv"
df = get_csv_from_url(url)
print("\nDATAFRAME:\n",df)

 