import numpy as np
import mltools.models as models

class LinearLS:
    #
    def __init__(self):
        self.model = None
    #
    def fit(self, X, Y):
        N, n = X.shape
        X1 = np.empty((N,n+1), np.double)
        X1[:,0] = 1
        X1[:,1:] = X

        Y1 = X1.T @ Y
        C = X1.T @ X1
        W = np.linalg.inv(C) @ Y1

        if self.model is None:
            self.model = models.LinearModel(n)            
        self.model.init_param(W)

class LinearWLS:
    #
    def __init__(self):
        self.model = None
    #
    def fit(self, X, Y, sample_weights=None):
        if sample_weights is not None and len(X) != len(sample_weights):
            raise ValueError('len(X) != len(sample_weights)')
        
        N, n = X.shape
        X1 = np.empty((N,n+1), np.double)
        X1[:,0] = 1
        X1[:,1:] = X

        if sample_weights is None:
            sample_weights = np.ones(N, np.double)

        V = np.diag(sample_weights)
        C = X1.T @ V @ X1
        Y1 = X1.T @ V @ Y
        W = np.linalg.inv(C) @ Y1

        if self.model is None:
            self.model = models.LinearModel(n)            
        self.model.init_param(W)

class IRLS:
    #
    def __init__(self, func, tol=1.0e-6, n_iter=100):
        self.func = func
        self.tol = tol
        self.n_iter = n_iter
    #
    def fit(self, X, Y):
        func = self.func
        tol = self.tol

        qvals = []
        
        est = LinearWLS()
        est.fit(X, Y)
        
        err = est.model.evaluate(X) - Y
        s = s_min = func.evaluate(err).sum()
        qvals.append(s)
        
        param_min = est.model.param.copy()

        is_stop = False
        for K in range(self.n_iter):
            s_prev = s

            weights = func.derivative_div_x(err)
            est.fit(X, Y, weights)
            
            err = est.model.evaluate(X) - Y
            s = func.evaluate(err).sum()
            qvals.append(s)

            if abs(s - s_prev) / (1 + abs(s_min)) < tol:
                is_stop = True

            if s < s_min:
                s_min = s
                param_min = est.model.param.copy()

            if is_stop:
                break

        est.model.param[:] = param_min
        self.model = est.model
        self.qvals = qvals
    #

class MIRLS:
    #
    def __init__(self, aggfunc, tol=1.0e-8, n_iter=300):
        self.aggfunc = aggfunc
        self.tol = tol
        self.n_iter = n_iter
    #
    def fit(self, X, Y):
        aggfunc = self.aggfunc
        tol = self.tol

        qvals = []

        mod = models.LinearModel(X.shape[1])
        mod.param = np.random.random(mod.n_param)
        est = LinearWLS()
        est.model = mod
        
        err = est.model.evaluate(X) - Y
        err2 = err * err
        s = s_min = aggfunc.evaluate(err2)
        qvals.append(s)
        
        param_min = est.model.param.copy()

        is_stop = False
        for K in range(self.n_iter):
            s_prev = s

            weights = aggfunc.gradient(err2)
            est.fit(X, Y, weights)
            
            err = est.model.evaluate(X) - Y
            err2 = err * err
            s = aggfunc.evaluate(err2)
            qvals.append(s)

            if abs(s - s_prev) / (1 + abs(s_min)) < tol:
                is_stop = True

            if s < s_min:
                s_min = s
                param_min = est.model.param.copy()

            if is_stop:
                break

        est.model.param[:] = param_min
        self.model = est.model
        self.qvals = qvals
    #
