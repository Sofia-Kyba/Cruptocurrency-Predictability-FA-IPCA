import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from ipca_prop.ipca import IPCARegressor
from sklearn.preprocessing import LabelEncoder

from scipy.stats import ttest_1samp
from matplotlib.ticker import PercentFormatter
from datetime import datetime

Kmax = 7
 
for SelectNoFactors in range(1,Kmax+1):
    
    sample = 'daily'
    
    print('Number of factors: K='+str(SelectNoFactors))
    SampleFreq = 'Daily'
    SampleFactor = 12*30
    SelectModelVersion = 'Restr'
        
    TESTPFs_path = '../Latex_Marginal_R2/'
    
    data_path = './../Data/'
    res_file = 'IPCAResults.pkl'
    results_path = '../RESULTS_'+sample+'/'
    input_file = 'cryptos_daily_social.csv'

    factors_file = 'riskfactors_daily_social.csv'
    ObsFacSet = ['mkt_vw','ahcc07_0_f','r30_0_f','max30_0_f','es95_90_0_f','coefvar_vol07_0_f','reddit_subscribers_f']


    with open(results_path+res_file, 'rb') as handle:
        IPCAResults = pickle.load(handle)
    
    CaseResults = IPCAResults['K='+str(SelectNoFactors)+'_'+SelectModelVersion]
    
    IPCAFactors = CaseResults[14].T
    IPCADates = CaseResults[15]
    IPCADates = IPCADates['DATE']
    IPCADates = IPCADates.unique()
    IPCADates = pd.DataFrame(data=IPCADates, columns=['Date'])
    Ordering = np.argsort(np.var(IPCAFactors ,axis=0))
    
    IPCAFactorsnames = ['F'+str(i+1) for i in range(SelectNoFactors)]
    IPCAFactors = pd.DataFrame(data=IPCAFactors[:,Ordering[::-1]], index=IPCADates['Date'], columns=IPCAFactorsnames)
    IPCAFactors.index.name = 'Date'

    APFactors = pd.read_csv(data_path+factors_file, sep=',')
    

    APFactors.DATE = pd.to_datetime(APFactors.DATE, dayfirst=True)
    APFactors['DATE'] = APFactors['DATE'].apply(lambda x: x.toordinal())
    APFactors.set_index(['DATE'], inplace=True)
    APFactors = APFactors.loc[APFactors.index.isin(IPCADates['Date']), :]

    Models = {'IPCA_'+str(SelectNoFactors): IPCAFactors,
              'ObsFac_'+str(SelectNoFactors): APFactors[ObsFacSet[:SelectNoFactors]]}
    

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
    
    fxdata = fxdata.loc[ (fxdata.DATE>=pd.Timestamp(2017, 9, 1)), :] #
    fxdata = fxdata.loc[ (fxdata.DATE<=pd.Timestamp(2023, 1, 1)), :] #
    
    fxdata.dropna(subset=['FX'], inplace=True)
    
    predvars_remove = predvars

    
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
    
    for col in predvars:
        if col in ['const']:
            continue
        fxdata[col] = fxdata[col].groupby(level='DATE').apply(Scaler_CS)
        print(col)
    
    fxdata['const'] = 1.0
    
    fxdata.reset_index(inplace=True)
    fxdata.dropna(axis=0, how='any', inplace=True)
    
    dates = fxdata['DATE'].unique()
    ipca_cols = fxdata.columns[3:]
    
    regr = IPCARegressor(n_factors=SelectNoFactors, intercept=False, iter_tol=10e-4, max_iter=800)
    TestPFs, _ , _ = regr._unpack_panel(fxdata.to_numpy())
    
    TestPFs = TestPFs.T
    PFnames = ['P'+str(i+1) for i in range(np.size(TestPFs, axis=1))]
    TestPFs = pd.DataFrame(columns=PFnames, index=dates, data=TestPFs)
    
    TestPFs.to_csv(TESTPFs_path+'managed_portfolios_'+sample+'.csv')
    
    # save Sharpe ratios of managed portfolios
    TestPFsSharpeRatios = pd.DataFrame(columns=predvars[:-1])
    
    temp_SR = TestPFs.mean().values[:-1]/(TestPFs.std().values[:-1])
    
    if SampleFreq == 'Daily':
        TestPFsSharpeRatios.loc['Sharpe'] = temp_SR*np.sqrt(12*30/SampleFactor)
    if SampleFreq == 'Weekly':
        TestPFsSharpeRatios.loc['Sharpe'] = temp_SR*np.sqrt(12*4/SampleFactor)
    TestPFsSharpeRatios.loc['Sharpe (daily)'] = temp_SR
    TestPFsSharpeRatios.loc['t-stat'] = np.sqrt(np.size(TestPFs, axis = 0))*temp_SR/np.sqrt(1+0.5*temp_SR**2)
    
    TestPFsSharpeRatios = TestPFsSharpeRatios.T
    # TestPFsSharpeRatios.to_csv(results_path+'APTest/APTest_Sharpe_Ratios.csv')
    
    # Volatility targeting
    if SampleFreq == 'Daily':
        VolTarget = 0.05*12*30/SampleFactor
        TestPFsVol = TestPFs.std()
    if SampleFreq == 'Weekly':
        VolTarget = 0.35*12*4/SampleFactor
        TestPFsVol = TestPFs.std()
    TestPFsLev = VolTarget/TestPFsVol
    TestPFs = TestPFs.multiply(TestPFsLev)
    # fix means
    TestPFs = TestPFs.multiply(np.sign(TestPFs.mean()))
    
    if SampleFreq == 'Daily':
        PFs_mean = TestPFs.mean()*(12*30)/SampleFactor
    if SampleFreq == 'Weekly':
        PFs_mean = TestPFs.mean()*(12*4)/SampleFactor
    PFs_mean.name = 'Mean'
    
    if SampleFreq == 'Daily':
        PFs_Sharpe = TestPFs.mean()*(12*30/SampleFactor)/(TestPFs.std()*np.sqrt(12*30/SampleFactor))
    if SampleFreq == 'Weekly':
        PFs_Sharpe = TestPFs.mean()*(12*4/SampleFactor)/(TestPFs.std()*np.sqrt(12*4/SampleFactor))
    PFs_Sharpe.loc['MeanSharpe'] = PFs_Sharpe.loc[PFnames].mean()
    PFs_Sharpe.loc['MedianSharpe'] = PFs_Sharpe.loc[PFnames].median()
    
    # (No-Instruments)
    ResultStore = dict()
    UnconditionalAlphas = pd.DataFrame(columns=predvars[:-1]+['Avg. Abs. Alpha'])
    
    for modelname in Models.keys():
        tempdf = pd.DataFrame(columns=['coeff', 'tvalues'])
        for PF in TestPFs.columns[:-1]:
            model = Models[modelname]
            tempdata = pd.merge(model, TestPFs[PF], left_index=True, right_index=True, how='inner')
    
            mod = sm.OLS(tempdata[PF],sm.add_constant(tempdata[model.columns]))
            res = mod.fit(cov_type='HAC',cov_kwds={'maxlags': 30})
            #res = mod.fit()
            tempdf.loc[PF] = [res.params['const'], res.tvalues['const']]
        print(np.sum(tempdf['tvalues'].abs()>2.00))
        ResultStore[modelname] = tempdf

        if SampleFreq == 'Daily':
            tempdf['coeff'] = tempdf['coeff']*(12*30)/SampleFactor
        if SampleFreq == 'Weekly':
            tempdf['coeff'] = tempdf['coeff']*(12*4)/SampleFactor
        tempdf = tempdf.merge(PFs_mean, left_index=True, right_index=True, how='inner')
        
        fig = plt.figure()
        plt.style.use('seaborn-whitegrid')

        # Insignificant Alpha
        plt.scatter(tempdf.loc[tempdf.tvalues.abs()<2.00, 'Mean'], tempdf.loc[tempdf.tvalues.abs()<2.00,'coeff'], linewidths=1.3, marker='o', facecolor='none', edgecolors='k')

        # Significant Alpha
        plt.scatter(tempdf.loc[tempdf.tvalues.abs()>2.00, 'Mean'], tempdf.loc[tempdf.tvalues.abs()>2.00,'coeff'], linewidths=1.3, marker='D', facecolor='k', edgecolors='k')
    
        ax = plt.gca()
        x = np.linspace(*ax.get_xlim())
        ax.plot(x, x, color='k', linewidth=1.0, linestyle='--')
        ax.plot(x, np.zeros(np.shape(x)), color='k', linewidth=1.0, linestyle='--')
        vals_y = ax.get_yticks()
        ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals_y])
        vals_x = ax.get_xticks()
        ax.set_xticklabels(['{:,.1%}'.format(x) for x in vals_x])
        plt.xlabel('Raw Return')
        plt.ylabel('Alpha')
        textstr = 'Avg. Abs. Alpha = %.3f%%' % (tempdf['coeff'].abs().mean()*100)
        
        x_legend = (vals_x[0]+vals_x[-1])/2
        y_legend = vals_y[-2]
        plt.text(x_legend, y_legend, textstr,
                 {'color': 'black', 'fontsize': 11, 'ha': 'center', 'va': 'center',
                  'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
        
        textstr = '' 
        for i in range(len(predvars)-1):
            if np.abs(tempdf.tvalues[i]) > 2.00:
                textstr = '\n'.join((textstr,predvars[i]))

        
        fig.set_size_inches(5, 5)
        plt.savefig(results_path+'APTest/APTest_Uncond_'+modelname+'.pdf')
        save_name_temp = modelname
            
        UnconditionalAlphas.loc[save_name_temp+'_alpha'] = tempdf.append({'coeff' : tempdf['coeff'].abs().mean()},  ignore_index=True).coeff.values
        UnconditionalAlphas.loc[save_name_temp+'_tstat'] = tempdf.append({'tvalues' : float('nan')},  ignore_index=True).tvalues.values

    UnconditionalAlphas = UnconditionalAlphas.T
    UnconditionalAlphas.to_csv(results_path+'APTest/APTest_Uncond_'+ modelname +'.csv')

    ResultStoreCond = dict()
    ConditionalAlphas = pd.DataFrame(columns=predvars[:-1]+['Avg. Abs. Alpha'])
    
    for modelname in Models.keys():
    
        if modelname == 'IPCA_'+str(SelectNoFactors):
            regr = IPCARegressor(n_factors=SelectNoFactors, intercept=False, iter_tol=10e-4, max_iter=800)
            regr.fit(Panel=fxdata.to_numpy())
    
            TestPFs = regr.X.T
    
            IPCAPFs = np.full(np.shape(TestPFs), np.nan)
            for t_i, t in enumerate(regr.dates):
                IPCAPFs[t_i, :] = np.squeeze(regr.W[:, :, t_i].dot(regr.Gamma_Est)\
                                .dot(regr.Factors_Est[:, t_i]))
        else:
            regr = IPCARegressor(n_factors=np.shape(Models[modelname])[1],
                                 intercept=False, iter_tol=10e-4, max_iter=800)
            temp = Models[modelname].copy()
            temp.reset_index(inplace=True)
            regr.fit(Panel=fxdata.loc[fxdata.DATE.isin(temp.DATE), :].to_numpy(), PSF=Models[modelname].to_numpy().T)
    
            TestPFs = regr.X.T
    
            IPCAPFs = np.full(np.shape(TestPFs), np.nan)
            for t_i, t in enumerate(regr.dates):
                IPCAPFs[t_i, :] = np.squeeze(regr.W[:, :, t_i].dot(regr.Gamma_Est)\
                                .dot(regr.Factors_Est[:, t_i]))

    
        dates_temp = Models[modelname].index
        
        # load actual and predicted returns
        TestPFs = pd.DataFrame(columns=PFnames, data=TestPFs, index=dates_temp)
        IPCAPFs = pd.DataFrame(columns=PFnames, data=IPCAPFs, index=dates_temp)

        # find leverage mutlipliers
        PFsVol = TestPFs.std()
        PFsLev = VolTarget/PFsVol

        TestPFs = TestPFs.multiply(PFsLev)
        IPCAPFs = IPCAPFs.multiply(PFsLev)

        IPCAPFs = IPCAPFs.multiply(np.sign(TestPFs.mean()))
        TestPFs = TestPFs.multiply(np.sign(TestPFs.mean()))
        
        Resid = TestPFs-IPCAPFs
        Resid = pd.DataFrame(data=Resid, columns=PFnames, index=dates_temp)
        tempdf = pd.DataFrame(columns=['coeff', 'tvalues'])
        for PF in Resid.columns[:-1]:
            mod = sm.OLS(Resid[PF], np.ones((len(Resid))))
            res = mod.fit(cov_type='HAC',cov_kwds={'maxlags': 30})
            tempdf.loc[PF] = [res.params['const'], res.tvalues['const']]
        ResultStoreCond[modelname] = tempdf

        if SampleFreq == 'Daily':
            tempdf['coeff'] = tempdf['coeff']*(12*30)/SampleFactor
        if SampleFreq == 'Weekly':
            tempdf['coeff'] = tempdf['coeff']*(12*4)/SampleFactor
        tempdf = tempdf.merge(PFs_mean, left_index=True, right_index=True, how='inner')
        
        fig = plt.figure()
        plt.style.use('seaborn-whitegrid')

        plt.scatter(tempdf.loc[tempdf.tvalues.abs()<2.00, 'Mean'], tempdf.loc[tempdf.tvalues.abs()<2.00,'coeff'], linewidths=1.3, marker='o', facecolor='none', edgecolors='k')
        plt.scatter(tempdf.loc[tempdf.tvalues.abs()>2.00, 'Mean'], tempdf.loc[tempdf.tvalues.abs()>2.00,'coeff'], linewidths=1.3, marker='D', facecolor='k', edgecolors='k')

        ax = plt.gca()
        x = np.linspace(*ax.get_xlim())
        ax.plot(x, x, color='k', linewidth=1.0, linestyle='--')
        ax.plot(x, np.zeros(np.shape(x)), color='k', linewidth=1.0, linestyle='--')
        vals_y = ax.get_yticks()
        ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals_y])
        vals_x = ax.get_xticks()
        ax.set_xticklabels(['{:,.1%}'.format(x) for x in vals_x])
        plt.xlabel('Raw Return')
        plt.ylabel('Alpha')
        textstr = 'Avg. Abs. Alpha = %.3f%%' % (tempdf['coeff'].abs().mean()*100)
        
        x_legend = (vals_x[0]+vals_x[-1])/2
        y_legend = vals_y[-2]
        plt.text(x_legend, y_legend, textstr,
                 {'color': 'black', 'fontsize': 11, 'ha': 'center', 'va': 'center',
                  'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
        
        textstr = ''
        for i in range(len(predvars)-1):
            if np.abs(tempdf.tvalues[i]) > 2.00:
                textstr = '\n'.join((textstr,predvars[i]))

         
        fig.set_size_inches(5, 5)
        fig.savefig(results_path+'APTest/APTest_Cond_'+modelname+'.pdf')
        save_name_temp = modelname
            
        ConditionalAlphas.loc[save_name_temp+'_alpha'] = tempdf.append({'coeff' : tempdf['coeff'].abs().mean()},  ignore_index=True).coeff.values
        ConditionalAlphas.loc[save_name_temp+'_tstat'] = tempdf.append({'tvalues' : float('nan')},  ignore_index=True).tvalues.values

    
    ConditionalAlphas = ConditionalAlphas.T
    ConditionalAlphas.to_csv(results_path+'APTest/APTest_Cond_'+ modelname +'.csv')
    
    
    
    
