import numpy as np

from mltools.funcs import Square
from mltools.lossfuncs import ErrorLoss
from mltools.risks import WRisk
from mltools.gda import GradientDescent

class MIRLS:

    def __init__(self, mod, agg_func, n_iter=100, tol=1.0e-8, h=0.1):
        self.agg_func = agg_func
        self.n_iter = n_iter
        self.tol = tol
        self.h = h

        self.risk_func = WMRisk(mod, ErrorLoss(Square()))
    #
    def fit(self, X, Y):
        risk_func = self.risk_func
        mod = risk_func.model

        L = self.risk_func.evaluate_losses(X, Y)

        u = self.agg_func.evaluate(L)
        risk_func.weights = self.agg_func.gradient(L)
        
        gd = GradientDescent(risk_func, h=self.h)

        rval = rval_min = u
        param_min = mod.param.copy()
        rvals = [rval]
        # print(rval)

        stop = False
        for k in range(self.n_iter):
            rval_prev = rval

            gd.fit(X, Y)

            L = self.risk_func.evaluate_losses(X, Y)
    
            u = self.agg_func.evaluate(L)
            risk_func.weights = self.agg_func.gradient(L)
            # print("weights:", risk_func2.weights)

            rval = u
            rvals.append(rval)
            # print(k, rval)

            if abs(rval - rval_prev) / (1 + abs(rval_min)) < self.tol:
                stop = True

            if rval < rval_min:
                rval_min = rval
                param_min = mod.param.copy()

            if stop:
                break
        
        mod.param[:] = param_min
        self.rvals = rvals
        self.K = k + 1

        
            