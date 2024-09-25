import numpy as np
import sys

class GradientDescent:
    #
    def __init__(self, risk, h=0.1, n_iter=1000, tol=1.0e-8):
        self.risk = risk
        self.h = h
        self.n_iter = n_iter
        self.tol = tol
        self.model = risk.model
    #
    def fit(self, X, Y):
        tol = self.tol
        h = self.h
        risk = self.risk
        risk_model = risk.model

        rval = rval_min = risk.evaluate(X, Y)
        param_min = risk_model.param.copy()
        rvals = [rval_min]

        flag = False
        for K in range(self.n_iter):
            rval_prev = rval

            G = risk.gradient(X, Y)
            # np.subtract(risk_model.param, h * G, out=risk_model.param)
            risk_model.param[:] = risk_model.param - h * G
            
            # print("model:", risk.model.param)
            
            rval = risk.evaluate(X, Y)
            rvals.append(rval)
                        
            if abs(rval_prev - rval) / (1 + abs(rval_min)) < tol:
                flag = True

            if rval < rval_min:
                rval_min = rval
                param_min = risk_model.param.copy()

            if flag:
                break
        
        self.rvals = rvals
        risk_model.param[:] = param_min
        self.K = K+1
