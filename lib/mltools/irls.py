import numpy as np

from mltools.funcs import Square
from mltools.lossfuncs import ErrorLoss
from mltools.risks import WRisk, Risk
from mltools.gda import GradientDescent

class IRLS:

    def __init__(self, mod, rho_func, n_iter=100, tol=1.0e-8, h=0.1):
        self.rho_func = rho_func
        self.n_iter = n_iter
        self.tol = tol
        self.h = h

        self.risk_func = Risk(mod, ErrorLoss(self.rho_func))
        self.risk_func2 = WRisk(mod, ErrorLoss(Square()))
    #
    def fit(self, X, Y):
        risk_func = self.risk_func
        risk_func2 = self.risk_func2
        mod = risk_func.model

        Yp = mod.evaluate_all(X)
        E = Yp - Y
        L = self.rho_func.evaluate(E)

        risk_func2.weights = self.rho_func.derivative_div_x(E)
        risk_func2.weights /= np.sum(risk_func2.weights) 
        # print("weights:", risk_func2.weights)
        
        gd = GradientDescent(risk_func2, h=self.h)

        rval = rval_min = np.mean(L)
        param_min = mod.param.copy()
        rvals = [rval]
        # print(rval)

        stop = False
        for k in range(self.n_iter):
            rval_prev = rval

            gd.fit(X, Y)

            Yp = mod.evaluate_all(X)
            E = Yp - Y
            L = self.rho_func.evaluate(E)
    
            risk_func2.weights = self.rho_func.derivative_div_x(E)
            risk_func2.weights /= np.sum(risk_func2.weights) 
            # print("weights:", risk_func2.weights)

            rval = np.mean(L)
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

        
            