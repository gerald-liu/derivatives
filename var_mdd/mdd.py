#%%
import pandas as pd

#%%
data = pd.read_excel('data.xlsx', sheet_name=None, header=None)

# %%
df_csi500 = data['CSI500']
df_csi500.columns = ['date', 'close']

# %%
df_csi500['roll_max'] = df_csi500['close'].rolling(window=120, min_periods=1).max()

# %%
df_csi500['mdd'] = df_csi500['close']/df_csi500['roll_max'] - 1

# %%
df_csi500 = df_csi500[df_csi500['date'] > '2014-01-04']
# %%
df_csi500['mdd'].min()

# %%
df_csi500['mdd'].idxmin()
# %%
df_csi500.loc[2598, :]
# %%
