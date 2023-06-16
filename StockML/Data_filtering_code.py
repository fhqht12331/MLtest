import pandas as pd

df = pd.read_csv('...\\LGDstock2.csv')
# print(df)

df.head()
df['data'].unique()
# print(df['data'].unique())
monthly = df['data'].str.endswith('15') 
print(df[monthly])

df_final = pd.concat(df[monthly],ignore_index=True)
print(df_final)

df[monthly].to_csv('...\\LGDstockPrice1.csv', encoding='euc-kr')