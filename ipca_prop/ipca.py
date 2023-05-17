from sklearn.linear_model import ElasticNet
from joblib import Parallel, delayed
from numba import jit
import numpy as np
import scipy as sp
import progressbar
import warnings
import time
import re
from copy import deepcopy

class IPCARegressor:

    def __init__(self, n_factors=1, intercept=False, max_iter=10000,
                 iter_tol=10e-6):

        if not isinstance(n_factors, int) or n_factors < 0:
            raise ValueError('n_factors must be an int greater / equal 1.')
        if not isinstance(intercept, bool):
            raise NotImplementedError('intercept must be  boolean')
        if not isinstance(iter_tol, float) or iter_tol >= 1:
            raise ValueError('Iteration tolerance must be smaller than 1.')

        params = locals()
        for k, v in params.items():
            if k != 'self':
                setattr(self, k, v)


    def fit(self, Panel=None, PSF=None, refit=False, alpha=1., l1_ratio=0.5, **kwargs):

        if refit:
            try:
                self.X
            except AttributeError:
                raise ValueError('Refit only possible after initial fit.')

        if Panel is None:
            raise ValueError('Must pass panel input data.')
        else:
            Panel = Panel[~np.any(np.isnan(Panel), axis=1)]

        if not refit:
            X, W, val_obs = self._unpack_panel(Panel)
        else:
            Panel, X, W, val_obs = self.Panel, self.X, self.W, self.val_obs

        if PSF is not None:
            if np.size(PSF, axis=1) != np.size(np.unique(Panel[:, 1])):
                raise ValueError("""Number of PSF observations must match
                                 number of unique dates in panel P""")
            self.has_PSF = True
            self.n_PSF = np.size(PSF, axis=0)
        else:
            self.has_PSF = False

        if self.has_PSF:
            if np.size(PSF, axis=0) == self.n_factors:
                print("""Note: The number of factors (n_factors) to be
                      estimated matches the number of
                      pre-specified factors. No additional factors
                      will be estimated. To estimate additional
                      factors increase n_factors.""")

        if self.intercept:
            self.n_factors_eff = self.n_factors + 1
            if PSF is not None:
                PSF = np.concatenate((PSF, np.ones((1, self.T))), axis=0)
            elif PSF is None:
                PSF = np.ones((1, self.T))
        else:
            self.n_factors_eff = self.n_factors

        if np.size(Panel, axis=1) < 4:
            raise ValueError("""Must provide at least one characteristic or constant""")

        self.PSFcase = True if self.has_PSF or self.intercept else False

        # run IPCA
        Gamma, Factors = self._fit_ipca(X, W, val_obs, Panel=Panel, PSF=PSF, **kwargs)

        # store estimates
        if self.PSFcase:
            if self.intercept and self.has_PSF:
                PSF = PSF
            elif self.intercept:
                PSF = np.ones((1, len(self.dates)))
            if Factors is not None:
                Factors = np.concatenate((Factors, PSF), axis=0)
            else:
                Factors = PSF

        self.Gamma_Est, self.Factors_Est = Gamma, Factors

        if not refit:
            self.Panel = Panel
            self.PSF = PSF
            self.X = X
            self.W = W
            self.val_obs = val_obs

        # goodness of Fit
        self.r2_total, self.r2_pred, self.r2_pred_cond, self.r2_total_x, self.r2_pred_x, self.r2_pred_cond_x, self.r2_ts_x, self.r2_cs_x, self.r2_rpe_x, self.IC = \
            self._R2_comps(Panel=Panel)

        return self.Gamma_Est, self.Factors_Est


    def predict(self, Panel=None, mean_factor=False, cond_mean_factor=False):

        if Panel is None:
            raise ValueError("""A panel of characteristics data must be
                              provided.""")

        if np.any(np.isnan(Panel)):
            raise ValueError("""Cannot contain missing observations / nan
                              values.""")
        N = np.size(Panel, axis=0)
        Ypred = np.full((N), np.nan)

        mean_Factors_Est = np.mean(self.Factors_Est, axis=1).reshape((-1, 1))
        if mean_factor:
            Ypred[:] = np.squeeze(Panel[:, 2:].dot(self.Gamma_Est)\
                .dot(mean_Factors_Est))

        elif cond_mean_factor:
            for t_i, t in enumerate(self.dates):
                if t_i < 12*2:
                    cond_mean_Factors_Est = np.mean(self.Factors_Est[:, :t_i+1], axis=1).reshape((-1, 1))
                else:
                    cond_mean_Factors_Est = np.mean(self.Factors_Est[:, t_i-12*2:t_i+1], axis=1).reshape((-1, 1))
                ix = (Panel[:, 1] == t)
                Ypred[ix] = np.squeeze(Panel[ix, 2:].dot(self.Gamma_Est)\
                    .dot(cond_mean_Factors_Est))

        else:
            for t_i, t in enumerate(self.dates):
                ix = (Panel[:, 1] == t)
                Ypred[ix] = np.squeeze(Panel[ix, 2:].dot(self.Gamma_Est)\
                    .dot(self.Factors_Est[:, t_i]))

        return Ypred
    

    def predict_bf(self, Panel=None):

        if Panel is None:
            raise ValueError("""A panel of characteristics data must be
                              provided.""")

        if np.any(np.isnan(Panel)):
            raise ValueError("""Cannot contain missing observations / nan
                              values.""")
        N = np.size(Panel, axis=0)
        Ypred = np.full((N), np.nan)

        if self.intercept:
            for t_i, t in enumerate(self.dates):
                ix = (Panel[:, 1] == t)
                Ypred[ix] = np.squeeze(Panel[ix, 2:].dot(self.Gamma_Est[:, :-1])\
                    .dot(self.Factors_Est[:-1, t_i]))
        else:
            for t_i, t in enumerate(self.dates):
                ix = (Panel[:, 1] == t)
                Ypred[ix] = np.squeeze(Panel[ix, 2:].dot(self.Gamma_Est)\
                    .dot(self.Factors_Est[:, t_i]))

        return Ypred

    def predict_alpha(self, Panel=None):

        if Panel is None:
            raise ValueError("""A panel of characteristics data must be
                              provided.""")

        if np.any(np.isnan(Panel)):
            raise ValueError("""Cannot contain missing observations / nan
                              values.""")

        if self.intercept is False:
            raise ValueError("Requires fitting a model with intercept first.")


        N = np.size(Panel, axis=0)
        Ypred = np.full((N), np.nan)


        for t_i, t in enumerate(self.dates):
            ix = (Panel[:, 1] == t)
            Ypred[ix] = np.squeeze(Panel[ix, 2:].dot(self.Gamma_Est[:, -1])\
                .dot(self.Factors_Est[-1, t_i]))

        return Ypred


    def BS_Walpha(self, ndraws=1000, blocksize=1, n_jobs=1, backend='loky'):

        if not self.intercept:
            raise ValueError('Need to fit model with intercept first.')

        Walpha = self.Gamma_Est[:, -1].T.dot(self.Gamma_Est[:, -1])

        d = np.full((self.L, self.T), np.nan)

        for t_i in range(self.T):
            d[:, t_i] = self.X[:, t_i]-self.W[:, :, t_i].dot(self.Gamma_Est)\
                .dot(self.Factors_Est[:, t_i])

        print("Starting Bootstrap...")
        Walpha_b = Parallel(n_jobs=n_jobs, backend=backend, verbose=20)(
            delayed(_BS_Walpha_sub_block)(self, n, d, blocksize) for n in range(ndraws))
        print("Done!")

        print(Walpha_b, Walpha)
        pval = np.sum(Walpha_b > Walpha)/ndraws
        return pval

    def BS_Wbeta(self, l, component=None, ndraws=1000, blocksize=1, n_jobs=1, backend='loky'):

        if self.PSFcase:
            raise ValueError('Need to fit model without intercept first.')

        if component is None:
            raise ValueError("Need to choose one of 'total', 'factor_x' where x=1,2,...")
        elif component == 'total':
            gamma_pos = slice(0,None,None)
        elif re.match('^factor_[0-9]+', component):
            gamma_pos = int(component[component.find('_')+1:])-1
        else:
            raise ValueError("component needs to be one of 'total', 'factor_x' where x=1,2,...")

        print("Running component: ", component)

        Wbeta_l = np.squeeze(self.Gamma_Est[l, gamma_pos].reshape((1, -1)).dot(self.Gamma_Est[l, gamma_pos].reshape((1, -1)).T))

        d = np.full((self.L, self.T), np.nan)
        for t_i, t in enumerate(self.dates):
            d[:, t_i] = self.X[:, t_i]-self.W[:, :, t_i].dot(self.Gamma_Est)\
                .dot(self.Factors_Est[:, t_i])

        print("Starting Bootstrap...")
        Wbeta_l_b = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_BS_Wbeta_sub_block)(self, n, d, l, gamma_pos, blocksize) for n in range(ndraws))
        print("Done!")

        pval = np.sum(Wbeta_l_b > Wbeta_l)/ndraws
        print(Wbeta_l_b, Wbeta_l)

        return pval


    def BS_Wdelta(self, ndraws=1000, blocksize=1, n_jobs=1, backend='loky'):

        if self.intercept:
            raise ValueError('Need to fit model without intercept first.')
        if not self.has_PSF:
            raise ValueError('Need to fit model with pre-specified factors first.')

        L, Ktilde = np.shape(self.Gamma_Est)
        K_PSF, _ = np.shape(self.PSF)
        K = Ktilde - K_PSF

        Wdelta = (self.Gamma_Est[:, -1].reshape((-1, 1), order="F")).T.dot(self.Gamma_Est[:, -1].reshape((-1, 1), order="F"))
        Wdelta = np.squeeze(Wdelta)

        d = np.full((self.L, self.T), np.nan)

        for t_i, t in enumerate(self.dates):
            d[:, t_i] = self.X[:, t_i]-self.W[:, :, t_i].dot(self.Gamma_Est)\
                .dot(self.Factors_Est[:, t_i])

        print("Starting Bootstrap...")
        Wdelta_b = Parallel(n_jobs=n_jobs, backend=backend, verbose=20)(
            delayed(_BS_Wdelta_sub_block)(self, n, d, blocksize) for n in range(ndraws))
        print("Done!")

        print(Wdelta_b, Wdelta)
        pval = np.sum(Wdelta_b > Wdelta)/ndraws
        return pval
    

    def predictOOS(self, Panel=None, mean_factor=False):

        if Panel is None:
            raise ValueError("""A panel of characteristics data must be
                              provided.""")

        if len(np.unique(Panel[:, 1])) > 1:
            raise ValueError('The panel must only have a single timestamp.')

        N = np.size(Panel, axis=0)
        Ypred = np.full((N), np.nan)

        Z, Y = Panel[:, 3:], Panel[:, 2]

        Numer = self.Gamma_Est.T.dot(Z.T).dot(Y)
        Denom = self.Gamma_Est.T.dot(Z.T).dot(Z).dot(self.Gamma_Est)
        Factor_OOS = np.linalg.solve(Denom, Numer.reshape((-1, 1)))

        if mean_factor:
            Ypred = np.squeeze(Z.dot(self.Gamma_Est)\
                    .dot(np.mean(self.Factors_Est, axis=1).reshape((-1, 1))))
        else:
            Ypred = Z.dot(self.Gamma_Est).dot(Factor_OOS)

        return Ypred


    def _unpack_panel(self, Panel):

        dates = np.unique(Panel[:, 1])
        ids = np.unique(Panel[:, 0])
        T = np.size(dates, axis=0)
        N = np.size(ids, axis=0)
        L = np.size(Panel, axis=1) - 3
        print('The panel dimensions are:')
        print('n_samples:', N, ', L:', L, ', T:', T)

        bar = progressbar.ProgressBar(maxval=T,
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ', progressbar.Percentage()])
        bar.start()
        X = np.full((L, T), np.nan)
        W = np.full((L, L, T), np.nan)
        val_obs = np.full((T), np.nan)
        for t_i, t in enumerate(dates):
            ixt = (Panel[:, 1] == t)
            val_obs[t_i] = np.sum(ixt)
            # Define characteristics weighted matrices
            X[:, t_i] = Panel[ixt, 3:].T.dot(Panel[ixt, 2])/val_obs[t_i]
            W[:, :, t_i] = Panel[ixt, 3:].T.dot(Panel[ixt, 3:])/val_obs[t_i]
            bar.update(t_i)
        bar.finish()

        self.ids, self.dates, self.T, self.N, self.L = ids, dates, T, N, L

        return X, W, val_obs


    def _fit_ipca(self, X, W, val_obs, Panel=None, PSF=None, quiet=False, **kwargs):

        Gamma_Old, s, v = np.linalg.svd(X)
        Gamma_Old = Gamma_Old[:, :self.n_factors_eff]
        s = s[:self.n_factors_eff]
        v = v[:self.n_factors_eff, :]
        Factor_Old = np.diag(s).dot(v)

        tol_current = 1

        iter = 0

        while((iter <= self.max_iter) and (tol_current > self.iter_tol)):

            Gamma_New, Factor_New = self._ALS_fit(Gamma_Old, W, X, val_obs, Panel=Panel, PSF=PSF, **kwargs)
            if self.PSFcase:
                tol_current = np.max(np.abs(Gamma_New - Gamma_Old))
            else:
                tol_current_G = np.max(np.abs(Gamma_New - Gamma_Old))
                tol_current_F = np.max(np.abs(Factor_New - Factor_Old))
                tol_current = max(tol_current_G, tol_current_F)

            Factor_Old, Gamma_Old = Factor_New, Gamma_New

            iter += 1
            if not quiet:
                print('Step', iter, '- Aggregate Update:', tol_current)

        if not quiet:
            print('-- Convergence Reached --')

        return Gamma_New, Factor_New


    def _ALS_fit(self, Gamma_Old, W, X, val_obs, Panel=None, PSF=None,
                 n_jobs=1, backend="loky", **kwargs):

        T = self.T

        if PSF is None:
            L, K = np.shape(Gamma_Old)
            Ktilde = K
        else:
            L, Ktilde = np.shape(Gamma_Old)
            K_PSF, _ = np.shape(PSF)
            K = Ktilde - K_PSF

        if K > 0:

            # no observed factors
            if PSF is None:
                if n_jobs > 1:
                    F_New = Parallel(n_jobs=n_jobs, backend=backend)(
                                delayed(_Ft_fit)(
                                    Gamma_Old, W[:,:,t], X[:,t])
                                for t in range(T))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t in range(T):
                        F_New[:,t] = _Ft_fit(Gamma_Old, W[:,:,t], X[:,t])

            # observed factors+latent factors case
            else:
                if n_jobs > 1:
                    F_New = Parallel(n_jobs=n_jobs, backend=backend)(
                                delayed(_Ft_PSF_fit)(
                                    Gamma_Old, W[:,:,t], X[:,t], PSF[:,t],
                                    K, Ktilde)
                                for t in range(T))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t in range(T):
                        F_New[:,t] = _Ft_PSF_fit(Gamma_Old, W[:,:,t], X[:,t],
                                                 PSF[:,t], K, Ktilde)

        else:
            F_New = None

        Gamma_New = _Gamma_portfolio_fit(F_New, X, W, val_obs, PSF, L, K,
                                         Ktilde, T)


        if PSF is not None and K>0:
            regbeta = np.linalg.lstsq(Gamma_New[:, :K], Gamma_New[:, K:])
            regbeta = regbeta[0]
            Gamma_New[:, K:] = Gamma_New[:, K:]-Gamma_New[:, :K].dot(regbeta)
            F_New += regbeta.dot(PSF)


        if K > 0:
            R1 = _numba_chol(Gamma_New[:, :K].T.dot(Gamma_New[:, :K])).T
            R2, _, _ = _numba_svd(R1.dot(F_New).dot(F_New.T).dot(R1.T))
            Gamma_New[:, :K] = _numba_lstsq(Gamma_New[:, :K].T,
                                            R1.T)[0].dot(R2)
            F_New = _numba_solve(R2, R1.dot(F_New))


        if K > 0:
            sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
            sg[sg == 0] = 1
            Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
            F_New = np.multiply(F_New, sg)

        return Gamma_New, F_New


    def _R2_comps(self, Panel=None):

        Ytrue = Panel[:, 2]

        # R2 total
        Ypred = self.predict(np.delete(Panel, 2, axis=1), mean_factor=False)
        r2_total = 1-np.nansum((Ypred-Ytrue)**2)/np.nansum(Ytrue**2)

        # R2 pred
        Ypred = self.predict(np.delete(Panel, 2, axis=1), mean_factor=True)
        r2_pred = 1-np.nansum((Ypred-Ytrue)**2)/np.nansum(Ytrue**2)

        Num_tot, Denom_tot = 0, 0
        Num_pred, Denom_pred = 0, 0

        mean_Factors_Est = np.mean(self.Factors_Est, axis=1).reshape((-1, 1))

        cond_mean_Factors_Est = self.Factors_Est.copy()
        for t_i, t in enumerate(self.dates):
            if t_i < 12*2:
                cond_mean_Factors_Est[:, t_i] = np.squeeze(np.mean(self.Factors_Est[:, :t_i+1], axis=1))
            else:
                cond_mean_Factors_Est[:, t_i] = np.squeeze(np.mean(self.Factors_Est[:, t_i-12*2:t_i+1], axis=1))

        for t_i, t in enumerate(self.dates):
            Ytrue = self.X[:, t_i]
            # R2 total
            Ypred = self.W[:, :, t_i].dot(self.Gamma_Est)\
                .dot(self.Factors_Est[:, t_i])
            Num_tot += (Ytrue-Ypred).T.dot((Ytrue-Ypred))
            Denom_tot += Ytrue.T.dot(Ytrue)

            # R2 pred
            Ypred = self.W[:, :, t_i].dot(self.Gamma_Est).dot(mean_Factors_Est)
            Ypred = np.squeeze(Ypred)
            Num_pred += (Ytrue-Ypred).T.dot((Ytrue-Ypred))
            Denom_pred += Ytrue.T.dot(Ytrue)
            

        r2_total_x = 1-Num_tot/Denom_tot
        r2_pred_x = 1-Num_pred/Denom_pred

        N = self.L
        V = 1/(N*self.T)*Num_tot
        k = self.n_factors

        IC = np.log(V) + k*(N+self.T)/(N*self.T)*np.log((N*self.T)/(N+self.T))


        return r2_total, r2_pred, r2_pred_cond, r2_total_x, r2_pred_x, r2_pred_cond_x, r2_ts_x, r2_cs_x, r2_rpe_x, IC

def _Ft_fit(Gamma_Old, W_t, X_t):

    m1 = Gamma_Old.T.dot(W_t).dot(Gamma_Old)
    m2 = Gamma_Old.T.dot(X_t)

    return np.squeeze(_numba_solve(m1, m2.reshape((-1, 1))))


def _Ft_PSF_fit(Gamma_Old, W_t, X_t, PSF_t, K, Ktilde):

    m1 = Gamma_Old[:,:K].T.dot(W_t).dot(Gamma_Old[:,:K])
    m2 = Gamma_Old[:,:K].T.dot(X_t)
    m2 -= Gamma_Old[:,:K].T.dot(W_t).dot(Gamma_Old[:,K:Ktilde]).dot(PSF_t)

    return np.squeeze(_numba_solve(m1, m2.reshape((-1, 1))))


def _Gamma_portfolio_fit(F_New, X, W, val_obs, PSF, L, K, Ktilde, T):

    Numer = _numba_full((L*Ktilde, 1), 0.0)
    Denom = _numba_full((L*Ktilde, L*Ktilde), 0.0)

    if PSF is None:
        for t in range(T):

            Numer += _numba_kron(X[:, t].reshape((-1, 1)),
                                 F_New[:, t].reshape((-1, 1)))\
                                 * val_obs[t]
            Denom += _numba_kron(W[:, :, t],
                                 F_New[:, t].reshape((-1, 1))
                                 .dot(F_New[:, t].reshape((1, -1)))) \
                                 * val_obs[t]

    elif K > 0:
        for t in range(T):
            Numer += _numba_kron(X[:, t].reshape((-1, 1)),
                                 np.vstack(
                                 (F_New[:, t].reshape((-1, 1)),
                                 PSF[:, t].reshape((-1, 1)))))\
                                 * val_obs[t]
            Denom_temp = np.vstack((F_New[:, t].reshape((-1, 1)),
                                   PSF[:, t].reshape((-1, 1))))
            Denom += _numba_kron(W[:, :, t], Denom_temp.dot(Denom_temp.T)
                                 * val_obs[t])

    else:
        for t in range(T):
            Numer += _numba_kron(X[:, t].reshape((-1, 1)),
                                 PSF[:, t].reshape((-1, 1)))\
                                 * val_obs[t]
            Denom += _numba_kron(W[:, :, t],
                                 PSF[:, t].reshape((-1, 1))
                                 .dot(PSF[:, t].reshape((-1, 1)).T))\
                                 * val_obs[t]

    Gamma_New = _numba_solve(Denom, Numer).reshape((L, Ktilde))

    return Gamma_New


def _Gamma_panel_fit(F_New, Panel, PSF, L, Ktilde, alpha, l1_ratio, **kwargs):

    if PSF is None:
        F = F_New
    else:
        if F_New is None:
            F = PSF
        else:
            F = np.vstack((F_New, PSF))
    F = F[:,np.unique(Panel[:,1], return_inverse=True)[1]]

    ZkF = np.hstack((F[k,:,None] * Panel[:,3:] for k in range(Ktilde)))

    if alpha:
        mod = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)
        mod.fit(ZkF, Panel[:,2])
        gamma = mod.coef_

    else:
        gamma = _numba_lstsq(ZkF, Panel[:,2])[0]

    gamma = gamma.reshape((Ktilde, L)).T

    return gamma


def _BS_Walpha_sub(model, n, d):
    X_b = np.full((model.L, model.T), np.nan)
    np.random.seed(n)
    Gamma = None

    # re-estimate unrestricted model
    while Gamma is None:
        try:
            for t in range(model.T):
                dof = 5
                tvar = dof / (dof-2)
                tstudent = (1/np.sqrt(tvar))*np.random.standard_t(dof)
                d_temp = tstudent*d[:,np.random.randint(0,high=model.T)]
                X_b[:, t] = model.W[:, :, t].dot(model.Gamma_Est[:, :-1])\
                    .dot(model.Factors_Est[:-1, t]) + d_temp

            Gamma, Factors = model._fit_ipca(X=X_b, W=model.W, val_obs=model.val_obs,
                                              PSF=model.PSF, quiet=True)

        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                            Observation discarded.")
            pass

    Walpha_b = Gamma[:, -1].T.dot(Gamma[:, -1])

    return Walpha_b

def _BS_Walpha_sub_block(model, n, d, blocksize):

    model = deepcopy(model)

    X_b = np.full((model.L, model.T), np.nan)
    np.random.seed(n)

    n_pf = np.shape(d)[0]

    block_len = blocksize

    n_blocks = np.ceil(np.size(d, axis=1)/block_len)
    last_block = np.size(d, axis=1)%block_len
    d_temp = d.copy()

    for b in range(int(n_blocks)):

        dof = 5
        tvar = dof / (dof-2)
        block_rv = (1/np.sqrt(tvar))*np.random.standard_t(dof)
        
        block_rv = np.random.standard_t(dof)

        if last_block == 0:
            rand_block = np.random.randint(0,high=n_blocks)
        else:
            rand_block = np.random.randint(0,high=n_blocks-1)

        if b < n_blocks-1:
            d_temp[:, b*block_len:(b+1)*block_len] = \
            d[:, block_len*rand_block:block_len*(rand_block+1)]*block_rv

        elif last_block > 0:
            d_temp[:, b*block_len:b*block_len+last_block] = \
            d[:, block_len*rand_block:block_len*rand_block+last_block]*block_rv

    for t in range(model.T):
        X_b[:, t] = model.W[:, :, t].dot(model.Gamma_Est[:, :-1])\
            .dot(model.Factors_Est[:-1, t]) + d_temp[:, t]

    Gamma = None
    while Gamma is None:
        try:

            Gamma, Factors = model._fit_ipca(X=X_b, W=model.W, val_obs=model.val_obs,
                                             PSF=model.PSF, quiet=True)
        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass

    Walpha_b = Gamma[:, -1].T.dot(Gamma[:, -1])

    return  Walpha_b

def _BS_Wbeta_sub_block(model, n, d, l, g, blocksize):
    X_b = np.full((model.L, model.T), np.nan)
    np.random.seed(n)

    Gamma_beta_l = np.copy(model.Gamma_Est)
    Gamma_beta_l[l, g] = 0
    Gamma = None
    n_pf = np.shape(d)[0]


    block_len = blocksize

    n_blocks = np.ceil(np.size(d, axis=1)/block_len)
    last_block = np.size(d, axis=1)%block_len
    d_temp = d.copy()

    for b in range(int(n_blocks)):

        dof = 5
        tvar = dof / (dof-2)
        block_rv = (1/np.sqrt(tvar))*np.random.standard_t(dof)

        if last_block == 0:
            rand_block = np.random.randint(0,high=n_blocks)
        else:
            rand_block = np.random.randint(0,high=n_blocks-1)

        if b < n_blocks-1:
            d_temp[:, b*block_len:(b+1)*block_len] = \
            d[:, block_len*rand_block:block_len*(rand_block+1)]*block_rv

        elif last_block > 0:
            d_temp[:, b*block_len:b*block_len+last_block] = \
            d[:, block_len*rand_block:block_len*rand_block+last_block]*block_rv

    while Gamma is None:
        try:
            for t in range(model.T):
                X_b[:, t] = model.W[:, :, t].dot(Gamma_beta_l)\
                    .dot(model.Factors_Est[:, t]) + d_temp[:, t]

            Gamma, Factors = model._fit_ipca(X=X_b, W=model.W, val_obs=model.val_obs,
                                             PSF=model.PSF, quiet=True)

        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass

    Wbeta_l_b = np.squeeze(Gamma[l, g].reshape((1, -1)).dot(Gamma[l, g].reshape((1, -1)).T))
    return float(Wbeta_l_b)

def _BS_Wdelta_sub_block(model, n, d, blocksize):
    X_b = np.full((model.L, model.T), np.nan)
    np.random.seed(n)

    L, Ktilde = np.shape(model.Gamma_Est)
    K_PSF, _ = np.shape(model.PSF)
    K = Ktilde - K_PSF
    n_pf = np.shape(d)[0]

    block_len = blocksize

    n_blocks = np.ceil(np.size(d, axis=1)/block_len)
    last_block = np.size(d, axis=1)%block_len
    d_temp = d.copy()

    for b in range(int(n_blocks)):
        dof = 5
        tvar = dof / (dof-2)
        block_rv = (1/np.sqrt(tvar))*np.random.standard_t(dof)


        if last_block == 0:
            rand_block = np.random.randint(0,high=n_blocks)
        else:
            rand_block = np.random.randint(0,high=n_blocks-1)

        if b < n_blocks-1:
            d_temp[:, b*block_len:(b+1)*block_len] = \
            d[:, block_len*rand_block:block_len*(rand_block+1)]*block_rv

        elif last_block > 0:
            d_temp[:, b*block_len:b*block_len+last_block] = \
            d[:, block_len*rand_block:block_len*rand_block+last_block]*block_rv


    Gamma_Est_CF = np.copy(model.Gamma_Est)

    Gamma_Est_CF[:, -1] = 0
    for t in range(model.T):
        X_b[:, t] = model.W[:, :, t].dot(Gamma_Est_CF)\
            .dot(model.Factors_Est[:, t]) + d_temp[:, t]

    Gamma = None
    while Gamma is None:
        try:
            Gamma, Factors = model._fit_ipca(X=X_b, W=model.W, val_obs=model.val_obs,
                                             PSF=model.PSF, quiet=False)
        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass

    Wdelta_b = (Gamma[:, -1].reshape((-1, 1), order="F")).T.dot(Gamma[:, -1].reshape((-1, 1), order="F"))

    return np.squeeze(Wdelta_b)


@jit(nopython=True)
def _numba_solve(m1, m2):
    return np.linalg.solve(np.ascontiguousarray(m1), np.ascontiguousarray(m2))

@jit(nopython=True)
def _numba_lstsq(m1, m2):
    return np.linalg.lstsq(np.ascontiguousarray(m1), np.ascontiguousarray(m2))

@jit(nopython=True)
def _numba_kron(m1, m2):
    return np.kron(np.ascontiguousarray(m1), np.ascontiguousarray(m2))

@jit(nopython=True)
def _numba_chol(m1):
    return np.linalg.cholesky(np.ascontiguousarray(m1))

@jit(nopython=True)
def _numba_svd(m1):
    return np.linalg.svd(np.ascontiguousarray(m1))

@jit(nopython=True)
def _numba_full(m1, m2):
    return np.full(m1, m2)
