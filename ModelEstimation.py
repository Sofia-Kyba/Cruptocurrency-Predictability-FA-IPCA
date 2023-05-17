import pandas as pd
import numpy as np
from ipca_prop.ipca import IPCARegressor
# from ipca import IPCARegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

import pickle
import os


SampleFreq = 'Monthly'
data_path = './../Data/'
results_path = '../RESULTS_test/'
input_file = 'cryptos_daily_social.csv'
Kmax = 7


fxdata = pd.read_csv(data_path+input_file, sep=',')
depvar = 'FX'
fxdata.dropna(subset=['LOCATION'], inplace=True)
fxdata.dropna(subset=['FX'], inplace=True)


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
fxdata = fxdata.loc[ (fxdata.DATE>=pd.Timestamp(2021, 9, 1)), :] #
fxdata = fxdata.loc[ (fxdata.DATE<=pd.Timestamp(2023, 1, 1)), :] #

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

def data_filtering(series):
    if len(series) < 100:
        series.loc[~series.isna()] = np.nan
    return series

fxdata.dropna(subset=['FX'], inplace=True)
 
# Scale variables in cross-section
def Scaler_CS(series):

    temp = series.loc[~series.isna()]
    try:

        temp2 = np.squeeze(temp.rank())
        n_cs = len(temp2)
        series.loc[~series.isna()] = temp2 - (1+n_cs)/2
        series.loc[series.isna()] = 0 # Replace missing obs with CS mean        
        series = series/(sum(series.loc[series > 0])/len(series))

    except ValueError:
        raise ValueError("CHECK DATA")

    return series

# check that the sum of positive (negative) weights is equal to 1 (-1)
def positive_sum(series):
    return sum(series.loc[series>0])/len(series)
def negative_sum(series):
    return sum(series.loc[series<0])/len(series)

fxdata_original = fxdata.copy()
fxdata_original.reset_index(inplace=True)

# Carry out scaling
for col in predvars:
    if col in ['const']:
        continue
    fxdata[col] = fxdata[col].groupby(level='DATE').apply(Scaler_CS)
    print(col,': ', np.mean(fxdata[col].groupby(level='DATE').apply(positive_sum)), ' ', np.mean(fxdata[col].groupby(level='DATE').apply(negative_sum)))

fxdata['const'] = 1.0

fxdata.reset_index(inplace=True)
fxdata.dropna(axis=0, how='any', inplace=True)

# IPCA Estimation


regr = IPCARegressor(n_factors=1, intercept=True, iter_tol=10e-4, max_iter=1000)

TestPFs, _ , _ = regr._unpack_panel(fxdata.to_numpy())

TestPFs = TestPFs.T
PFnames = ['P'+str(i+1) for i in range(np.size(TestPFs, axis=1))]
PFnames = predvars
TestPFs = pd.DataFrame(columns=PFnames, index=dates.DATE, data=TestPFs)
TestPFs.to_csv(results_path+'APTest/_APTest_managed_portfolios.csv')

# save Sharpe ratios of managed portfolios
TestPFsSharpeRatios = pd.DataFrame(columns=predvars[:-1])

temp_SR = TestPFs.mean().values[:-1]/(TestPFs.std().values[:-1])
TestPFsSharpeRatios.loc['Sharpe (daily)'] = temp_SR
TestPFsSharpeRatios.loc['t-stat'] = np.sqrt(np.size(TestPFs, axis = 0))*temp_SR/np.sqrt(1+0.5*temp_SR**2)

TestPFsSharpeRatios = TestPFsSharpeRatios.T
TestPFsSharpeRatios.to_csv(results_path+'APTest/_APTest_Sharpe_Ratios.csv')

# Unrestricted Case: With intercept
first_fit = True
for i in range(-1, Kmax):
    print(i+1)
    casename = 'K='+str(i+1)+'_Unrestr'
    if first_fit:
        regr.n_factors = i+1
        regr.fit(Panel=fxdata.to_numpy(), refit=False)
        first_fit = False
    else:
        regr.n_factors = i+1
        regr.fit(Panel=fxdata.to_numpy(), refit=True)

    # Prepare data for ex-post binning
    fxdata_aux = fxdata[['LOCATION', 'DATE']+[depvar]+predvars].copy()

    # Compute fitted values
    varlist = ['LOCATION', 'DATE']+[depvar]+predvars
    fxdata_aux[depvar[0]+'_hat'] = regr.predict(np.delete(fxdata_aux[varlist].to_numpy(), 2, axis=1),
                                           mean_factor=False)
    fxdata_aux[depvar[0]+'_hat_pred'] = regr.predict(np.delete(fxdata_aux[varlist].to_numpy(), 2, axis=1),
                                           mean_factor=True)

    # Compute Alphas
    fxdata_aux[depvar+'_hat_bf'] = regr.predict_bf(np.delete(fxdata_aux[varlist].to_numpy(), 2, axis=1))
    fxdata_aux[depvar+'_alpha'] = fxdata_aux[depvar] - fxdata_aux[depvar+'_hat_bf']
    
    # Compute squared residuals and returns
    fxdata_aux[depvar+'_resid2']  = (fxdata_aux[depvar] - fxdata_aux[depvar[0]+'_hat'])**2
    fxdata_aux[depvar+'_ret2'] = (fxdata_aux[depvar])**2


    ResultStore[casename] = [float(regr.r2_total),
                             float(regr.r2_pred),
                             float(regr.r2_total_x),
                             float(regr.r2_pred_x),
                             float(regr.IC),
                             regr.Gamma_Est,
                             regr.Factors_Est,
                             fxdata_aux]


# Restricted Case: No intercept
for i in range(Kmax):
    print(i+1)
    casename = 'K='+str(i+1)+'_Restr'

    regr.n_factors = i+1
    regr.intercept = False
    regr.fit(Panel=fxdata.to_numpy(), refit=True)

    # Prepare data for ex-post binning
    fxdata_aux = fxdata[['LOCATION', 'DATE']+[depvar]+predvars].copy()

    # Compute fitted values
    varlist = ['LOCATION', 'DATE']+[depvar]+predvars
    fxdata_aux[depvar[0]+'_hat'] = regr.predict(np.delete(fxdata_aux[varlist].to_numpy(), 2, axis=1),
                                           mean_factor=False)
    fxdata_aux[depvar[0]+'_hat_pred'] = regr.predict(np.delete(fxdata_aux[varlist].to_numpy(), 2, axis=1),
                                           mean_factor=True)
    
    # Compute alphas
    fxdata_aux[depvar+'_hat_bf'] = regr.predict_bf(np.delete(fxdata_aux[varlist].to_numpy(), 2, axis=1))
    fxdata_aux[depvar+'_alpha'] = fxdata_aux[depvar] - fxdata_aux[depvar+'_hat_bf']

    # Compute squared residuals and returns
    fxdata_aux[depvar+'_resid2']  = (fxdata_aux[depvar] - fxdata_aux[depvar[0]+'_hat'])**2
    fxdata_aux[depvar+'_ret2'] = (fxdata_aux[depvar])**2
    
    ResultStore[casename] = [float(regr.r2_total),
                             float(regr.r2_pred),
                             float(regr.r2_total_x),
                             float(regr.r2_pred_x),
                             float(regr.IC),
                             regr.Gamma_Est,
                             regr.Factors_Est,
                             fxdata_aux]


with open(results_path+'IPCAResults.pkl', 'wb') as handle:
    ExistResults = pickle.dump(ResultStore,handle, protocol=2)

R2_Results = pd.DataFrame(columns=['nfactors', 'intercept', 
                                   'r2_tot', 'r2_pred',
                                   'r2_tot_x', 'r2_pred_x',
                                   'IC'])
for case in ResultStore.keys():
    print(case)
    temp = ResultStore[case]
    if 'Unrestr' in case:
        intercept_bool = 'yes'
    else:
        intercept_bool = 'no'
    k = int(case[2:3])

    R2_Results = R2_Results.append({'nfactors': k,
                                    'intercept': intercept_bool,
                                    'r2_tot': temp[0],
                                    'r2_pred': temp[1],
                                    'r2_tot_x': temp[2],
                                    'r2_pred_x': temp[3],
                                    'IC': temp[4]}, ignore_index=True)

    Factors = ResultStore[case][14]
    columns = ['F'+str(i+1) for i in range(k)]
    if 'Unrestr' in case:
        columns += ['Intercept']
    Factors = pd.DataFrame(data=Factors.T,
                           columns=columns, 
                           index=np.squeeze(dates.values))
    Factors.to_csv(results_path+'FACTORS/FactorEst_'+case+'.csv')

    Gamma = ResultStore[case][13]
    Gamma = pd.DataFrame(data=Gamma,
                         columns=columns,
                         index=predvars)
    Gamma.to_csv(results_path+'GAMMA/GammaEst_'+case+'.csv')

    fxdata_hat = ResultStore[case][15].copy()
    
    data_names = fxdata_original.columns[:-1]
    fxdata_hat[data_names] = fxdata_original[data_names].copy()
    
    fxdata_hat['LOCATION'] = loc_encoder.inverse_transform(fxdata_hat['LOCATION'])
    fxdata_hat['DATE'] = fxdata_hat['DATE'].apply(lambda x: datetime.fromordinal(x))
    fxdata_hat.to_csv(results_path+'FittedValPanel_'+case+'.csv')

R2_Results.to_csv(results_path+'R2_Summary.csv')
