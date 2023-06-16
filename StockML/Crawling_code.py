import pandas as pd

stock_code = pd.read_excel('...\\stock_sc_DB.xlsx',sheet_name='Sheet1',converters={'종목코드':str})
stock_code = stock_code[['종목코드','종목명']]
# print(stock_code)

import requests
from bs4 import BeautifulSoup
import pandas as pd
import xmltodict
import json

def get_daily_stock_price(stockCode, name, count):

    url = f'https://fchart.stock.naver.com/sise.nhn?symbol={stockCode}&timeframe=day&count={count}&requestType=0'
    rs = requests.get(url)
    dt = xmltodict.parse(rs.text)
    js = json.dumps(dt, indent=4)
    js = json.loads(js)
    
    data = pd.json_normalize(js['protocol']['chartdata']['item'])
    df = data['@data'].str.split('|',expand=True)
    df.columns = ['data','open','high','low','close','Volume']
    df['name'] = str(name) 
    
    return df

tmp=[]
for index, row in stock_code.iterrows():
    tmp.append(get_daily_stock_price(row['종목코드'],row['종목명'],'2500'))
print(tmp)

df_final = pd.concat(tmp,ignore_index=True)
print(df_final)

df_final.to_csv('...\\LGDstock.csv', encoding='euc-kr')