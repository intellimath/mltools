import numpy as np
import sys

class AggFunc:
    #
    def evaluate(self, X):
        pass
    #
    def gradient(self, X):
        pass
    #
    def weights(self, X):
        return self.gradient(X)

np_full = np.full
np_mean = np.mean
np_sum = np.sum
np_dot = np.dot
np_putmask = np.putmask

class ArithMean(AggFunc):
    #
    def evaluate(self, X):
        return X.mean()
    #
    def gradient(self, X):
        n = len(X)
        return np.full(n, 1.0/n)

class RArithSum(AggFunc):
    #
    def __init__(self, func):
        self.func = func
    #
    def evaluate(self, X):
        return self.func.evaluate(X).sum()
    #
    def gradient(self, X):
        return self.func.derivative(X)
    #
    def weights(self, X):
        return self.func.derivative_div_x(X)

class MMean(AggFunc):
    #
    def __init__(self, rho_func, n_iter=1000, tol=1.0e-8):
        self.rho_func = rho_func
        self.n_iter = n_iter
        self.tol = tol
        self.u = None
    #
    def evaluate(self, X):
        tol = self.tol
        rho_func = self.rho_func
        
        u = u_min = X.mean()
        pval = pval_min = rho_func.evaluate(X - u).mean()
        
        pvals = [pval]
        
        for k in range(self.n_iter):
            pval_prev = pval
            u_prev = u
            
            V = rho_func.derivative_div_x(X - u)
            V /= V.sum()
            
            u = (V @ X)
            
            pval = rho_func.evaluate(X - u).mean()
            pvals.append(pval)
            
            if pval < pval_min:
                pval_min = pval
                u_min = u
                
            if abs(pval - pval_prev) / (1 + abs(pval_min)) < self.tol:
                break
        
        self.u = u_min
        self.pvals = pvals
        return u_min
    
    def gradient(self, X):
        if self.u is None:
            u = self.evaluate(X)
        else:
            u = self.u
            
        R = self.rho_func.derivative2(X - u)
        R /= R.sum()

        self.u = None 
        return R    
    
class CMMean(AggFunc):
    #
    def __init__(self, rho_func, n_iter=1000, tol=1.0e-8):
        self.rho_func = rho_func
        self.agg = MMean(rho_func)
        self.u = None
    #
    def evaluate(self, X):
        Y = X.copy()
        u = self.agg.evaluate(X)
        Y[X > u] = u
        self.u = u
        return Y.mean()
    #
    def gradient(self, X):
        N = len(X)
        if self.u is None:
            u = self.u = self.agg.evaluate(X)
        else:
            u = self.u

        G = self.agg.gradient(X)
        
        q = (X >= u).sum()
        G *= q
        
        G[X < u] += 1
        G /= N
        
        self.u = None
        return G
