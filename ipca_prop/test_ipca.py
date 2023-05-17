import pytest
import numpy as np
from sklearn.utils.testing import assert_raises
from statsmodels.datasets import grunfeld
import time

from ipca import IPCARegressor


# Test Construction Errors
@pytest.mark.fast_test
def test_construction_errors():
    assert_raises(ValueError, IPCARegressor, n_factors=-1)
    assert_raises(NotImplementedError, IPCARegressor, intercept='jabberwocky')
    assert_raises(ValueError, IPCARegressor, iter_tol=2)


# create test data and run package
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)

N = len(np.unique(data.firm))
ID = dict(zip(np.unique(data.firm).tolist(), np.arange(1, N+1)+5))
data.firm = data.firm.apply(lambda x: ID[x])

data = data[['firm', 'year', 'invest', 'value', 'capital']]

data = data.to_numpy()
PSF = np.random.randn(len(np.unique(data[:, 1])), 2)
PSF = PSF.reshape((2, -1))


regr = IPCARegressor(n_factors=1, intercept=False)
Gamma_New, Factor_New = regr.fit(Panel=data)
print('R2total', regr.r2_total)
print('R2pred', regr.r2_pred)
print('R2total_x', regr.r2_total_x)
print('R2pred_x', regr.r2_pred_x)
print(Gamma_New)
print(Factor_New)

regr = IPCARegressor(n_factors=1, intercept=True)
Gamma_New, Factor_New = regr.fit(Panel=data)
print('R2total', regr.r2_total)
print('R2pred', regr.r2_pred)
print('R2total_x', regr.r2_total_x)
print('R2pred_x', regr.r2_pred_x)


data_x = np.delete(data, 2, axis=1)
Ypred = regr.predict(Panel=data_x)

regr.n_factors = 2
regr.intercept = False
Gamma_New, Factor_New = regr.fit(Panel=data, refit=True)

data_refit = data[data[:, 1] != 1954, :]
Gamma_New, Factor_New = regr.fit(Panel=data_refit, refit=False)

PSF = np.random.randn(len(np.unique(data[:, 1])), 1)
PSF = PSF.reshape((1, -1))
regr = IPCARegressor(n_factors=2, intercept=False)
Gamma_New, Factor_New = regr.fit(Panel=data, PSF=PSF, refit=False)

PSF = np.random.randn(len(np.unique(data[:, 1])), 2)
PSF = PSF.reshape((2, -1))
regr = IPCARegressor(n_factors=2, intercept=False)
Gamma_New, Factor_New = regr.fit(Panel=data, PSF=PSF, refit=False)

# nan observations
regr = IPCARegressor(n_factors=1, intercept=True)
data_nan = data.copy()
data_nan[10:30, 2:] = np.nan
Gamma_New, Factor_New = regr.fit(Panel=data_nan)

# missing observations
regr = IPCARegressor(n_factors=1, intercept=True)
data_missing = data.copy()
data_missing = data_missing[:-10, :]
Gamma_New, Factor_New = regr.fit(Panel=data_missing)


# in-sample data
data_IS = data[data[:, 1] != 1954, :]
# out-of-sample data
data_OOS = data[data[:, 1] == 1954, :]

regr.fit(Panel=data_IS)
Ypred = regr.predictOOS(Panel=data_OOS, mean_factor=True)

regr = IPCARegressor(n_factors=1, intercept=True)
Gamma_New, Factor_New = regr.fit(Panel=data)
pval = regr.BS_Walpha(ndraws=10)
print('p-value', pval)

regr = IPCARegressor(n_factors=1, intercept=False)
Gamma_New, Factor_New = regr.fit(Panel=data)
pval = regr.BS_Wbeta([0, 1], ndraws=10, n_jobs=-1)
print('p-value', pval)
