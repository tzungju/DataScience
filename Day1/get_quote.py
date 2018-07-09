import requests
from io import StringIO
import pandas as pd
import numpy as np
datestr = '20180131'
r = requests.post('http://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date=' + datestr + '&type=ALL')
df = pd.read_csv(StringIO("\n".join([i.translate({ord(c): None for c in ' '}) 
    for i in r.text.split('\n') 
    if len(i.split('",')) == 17 and i[0] != '='])), header=0)
  
# 選擇 本益比 < 15 的所有股票  
df_filter = df[(pd.to_numeric(df['本益比'], errors='coerce') < 15) & (pd.to_numeric(df['本益比'], errors='coerce') > 0)]
print(df_filter[['證券名稱','本益比']])
    