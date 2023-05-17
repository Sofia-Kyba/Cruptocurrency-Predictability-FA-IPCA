import pandas as pd
import numpy as np
from ipca_prop.ipca import IPCARegressor
# from ipca import IPCARegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import datetime

import pickle
import os

SampleFreq = 'Monthly'
data_path = './../Data/'
results_path = '../RESULTS_daily/'
input_file = 'cryptos_daily_social.csv'
output_file ='BootstrapResult_Wbeta_K=4_joint_Block=1.txt'


fxdata = pd.read_csv(data_path+input_file, sep=',')
depvar = 'FX'

fxdata.dropna(subset=['LOCATION'], inplace=True)


predvars     = ['reddit_posts_per_day', 'fb_likes', 'fb_talking_about', 'twitter_followers', 'reddit_subscribers', 'reddit_active_users', 'transaction_count', 'average_transaction_value', 'volume', 'bidask', 'marketcap', 'bm', 'turnover', 'volshockSTD30', 'capm_alpha', 'capm_beta', 'idio_vol', 'r30_14', 'r180_60', 'new_addresses07_0', 'active_addresses07_0', 'max07_0', 'max30_0', 'r07_0', 'r30_0', 'skew07_0', 'skew30_0', 'var95_90_0', 'es95_90_0', 'yzvol07_0', 'yzvol30_0', 'coefvar_vol07_0', 'coefvar_vol30_0', 'coefvar_to07_0', 'coefvar_to30_0', 'volume07_0', 'volume30_0', 'turnover07_0', 'turnover30_0', 'ahcc07_0', 'ahcc30_0', 'realahcc_volume07_0', 'realahcc_volume30_0']
predvars_lag = ['reddit_posts_per_day', 'fb_likes', 'fb_talking_about', 'twitter_followers', 'reddit_subscribers', 'reddit_active_users', 'transaction_count', 'average_transaction_value', 'volume', 'bidask', 'marketcap', 'bm', 'turnover', 'volshockSTD30', 'capm_alpha', 'capm_beta', 'idio_vol', 'r30_14', 'r180_60', 'new_addresses07_0', 'active_addresses07_0', 'max07_0', 'max30_0', 'r07_0', 'r30_0', 'skew07_0', 'skew30_0', 'var95_90_0', 'es95_90_0', 'yzvol07_0', 'yzvol30_0', 'coefvar_vol07_0', 'coefvar_vol30_0', 'coefvar_to07_0', 'coefvar_to30_0', 'volume07_0', 'volume30_0', 'turnover07_0', 'turnover30_0', 'ahcc07_0', 'ahcc30_0', 'realahcc_volume07_0', 'realahcc_volume30_0', 'r07_0', 'r30_0']


loc_encoder = LabelEncoder()
fxdata['LOCATION'] = loc_encoder.fit_transform(fxdata['LOCATION'])

fxdata.drop(['DATE'], axis=1, inplace=True)
fxdata.rename(index=str, columns={'TIME_M': 'DATE'}, inplace=True)
fxdata['DATE'] = pd.to_datetime(fxdata['DATE'], dayfirst=True) # + pd.offsets.MonthEnd(0)

for col in predvars_lag:
    fxdata[col] = fxdata.groupby(['LOCATION'])[col].shift(+1)
    print(['Lagging observations in ', col])

# remove the first observations for missing momentum characteristics
fxdata = fxdata.loc[ (fxdata.DATE>=pd.Timestamp(2017, 9, 1)), :]
fxdata = fxdata.loc[ (fxdata.DATE<=pd.Timestamp(2023, 1, 1)), :]

# Remove obs with missing dependent variable
fxdata.dropna(subset=['FX'], inplace=True)

# Drop NaN in predictors
predvars_remove = predvars
predvars_remove = []

for col in predvars_remove:
    fxdata.dropna(subset=[col], inplace=True)
    print(['Dropping NaN in ', col])

dates = fxdata['DATE'].unique()
dates = pd.DataFrame(dates, columns=['DATE'])
dates = dates.sort_values(by='DATE')
fxdata['DATE'] = fxdata['DATE'].apply(lambda x: x.toordinal())

fxdata.set_index(['LOCATION', 'DATE'], inplace=True)
ResultStore = dict()

fxdata['const'] = 1
predvars.append('const')

fxdata = fxdata.loc[:, ['FX']+predvars]

def Scaler_CS(series):

    temp = series.loc[~series.isna()]
    try:

        temp2 = np.squeeze(temp.rank())
        n_cs = len(temp2)
        series.loc[~series.isna()] = temp2 - (1+n_cs)/2
        series.loc[series.isna()] = 0
        series = series/(sum(series.loc[series > 0])/len(series))

    except ValueError:
        raise ValueError("CHECK DATA")

    return series

for col in predvars:
    if col in ['const']:
        continue
    fxdata[col] = fxdata[col].groupby(level='DATE').apply(Scaler_CS)
    print(col)

fxdata['const'] = 1.0

fxdata.reset_index(inplace=True)
fxdata.dropna(axis=0, how='any', inplace=True)


regr = IPCARegressor(n_factors=4, intercept=False, iter_tol=10e-4, max_iter=1000)

regr.fit(Panel=fxdata.to_numpy(), refit=False)


ResultStore = dict()

for l_no, l in enumerate(predvars):
    # Compute Bootstrap p-values
        ResultStore[l] = regr.BS_Wbeta(l_no, 'total', ndraws=100, n_jobs=-1, blocksize=1, backend='loky')
        print(ResultStore)


log = open(results_path+output_file, "a+")
print("Bootstrap W-Beta", file=log)
print(datetime.datetime.now(),file=log)
print(ResultStore, file = log)
log.close()




