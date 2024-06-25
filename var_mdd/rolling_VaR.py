import pandas as pd
from scipy.stats import norm

data = pd.read_excel('data.xlsx', sheet_name=None, header=None)
VaR_window = 119

def VaR(df: pd.DataFrame, window):
    df.columns = ['date', 'close']

    df['close-lag'] = df['close'].shift(window)
    df['roll_ret'] = df['close']/df['close-lag'] - 1

    df = df[df['date'] > '2014-01-04']

    # print(df.loc[df['roll_ret'].idxmin(), :])
    
    list_out = [
        df['roll_ret'].min(),
        df['roll_ret'].quantile(0.001),
        df['roll_ret'].quantile(0.005),
        df['roll_ret'].quantile(0.01)
    ]

    return list_out

df_out = pd.DataFrame(columns=['Name', 'Min', '99.9%', '99.5%', '99%'])

for name, df in data.items():
    row = [name] + VaR(df, window=VaR_window)
    df_out.loc[len(df_out)] = row

print(df_out)
