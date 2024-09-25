import numpy as np
from math import sqrt

np_array = np.array
np_fromiter = np.fromiter
np_dot = np.dot
np_empty = np.empty
np_zeros = np.zeros
np_ones = np.ones
np_hstack = np.hstack

class Model:
    """
    param: вектор параметров модели
    n_param: длина вектора параметров
    n_input: длина вектора входов модели
    """
    #
    def evaluate_one(self, Xk):
        raise NotImplemented
    #
    def gradient_one(self, Xk):
        raise NotImplemented
    #
    def gradient_x_one(self, Xk):
        raise NotImplemented
    #
    def ievaluate(self, X):
        evaluate_one = self.evaluate_one
        for Xk in X:
            yield evaluate_one(Xk)
    #
    def evaluate(self, X):
        return np.fromiter(self.ievaluate(X), np.double, len(X))
    #
    def igradient(self, X):
        gradient_one = self.gradient_one
        for Xk in X:
            yield gradient_one(Xk)
    #
    def gradient(self, X):
        rows = tuple(self.igradient(X))
        return np.vstack(rows)

class LinearModel(Model):
    #
    def __init__(self, n):
        self.n_input = n
        self.n_param = n + 1
        self.param = np.zeros(self.n_param, np.double)
    #
    def evaluate_one(self, Xk):
        return self.param[0] + (self.param[1:] @ Xk)
    #
    def ievaluate(self, X):
        p0 = self.param[0]
        pp = self.param[1:]
        for Xk in X:
            yield p0 + pp @ Xk
    #
    def gradient_one(self, Xk):
        G = np.empty(self.n_param, np.double)
        G[0] = 1.
        G[1:] = Xk
        return G
    #
    def gradient(self, X):
        N = X.shape[0]
        G = np.empty((N, self.n_param), np.double)
        G[:,0] = 1.
        G[:, 1:] = X
        return G
    #
    def gradient_x(self, Xk):
        return self.param[1:].copy()
    #
    def evaluate(self, X):
        N = X.shape[0]
        X1 = np.empty((N, self.n_param), np.double)
        X1[:,0] = 1
        X1[:,1:] = X
        Y = X1 @ self.param
        return Y

class SigmaNeuronModel(Model):
    #
    def __init__(self, outfunc, n):
        self.outfunc = outfunc
        self.n_param = n + 1
        self.n_input = n
        self.param = np.zeros(self.n_param, np.double)
    #
    def evaluate_one(self, Xk):
        return self.outfunc.evaluate(self.param[0] + (self.param[1:] @ Xk))
    #
    def evaluate(self, X):
        N = X.shape[0]
        X1 = np.empty((N, self.n_input+1), np.double)
        X1[:,0] = 1
        X1[:,1:] = X
        Y = X1 @ self.param
        return self.outfunc.evaluate(Y)
    #
    def gradient(self, Xk):
        N = X.shape[0]
        X1 = np.empty((N, self.n_input+1), np.double)
        X1[:,0] = 1
        X1[:,1:] = X
        
        S = X1 @ self.param

        D = self.outfunc.derivative(S)
            
        G = X1 * D[:,None]
        
        return G
    #
    def gradient_x(self, X):
        N = X.shape[0]
        X1 = np.empty((N, self.n_input+1), np.double)
        X1[:,0] = 1
        X1[:,1:] = X
        
        S = X1 @ self.param
        D = self.outfunc.derivative(S)
        
        P = np.repeat(self.param[None,:], N)
        R = P * D[:,None]
        return R

class SimpleNN(Model):
    #
    def __init__(self, outfunc, n_input, n_hidden=0, add_root=True):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.outfunc = outfunc
        self.hidden = []
        self.n_hidden = 0
        if n_hidden > 0:
            for i in range(n_hidden):
                self.add_hidden()
            if add_root:
                self.add_root()
    #
    def add_hidden(self):
        self.hidden.append(SigmaNeuronModel(self.outfunc, self.n_input))
        self.n_hidden += 1
    #
    def add_root(self):
        self.root = LinearModel(self.n_hidden)
        self.n_param = self.root.n_param + sum([mod.n_param for mod in self.hidden])
        self.param = np.empty(self.n_param, 'd')
        self.root.param = self.param[:self.root.n_param]
        m = self.root.n_param
        for mod in self.hidden:
            mod.param = self.param[m : m+mod.n_param]
            m += mod.n_param
    #
    def evaluate(self, Xk):
        U = np.fromiter(
            (mod.evaluate(Xk) for mod in self.hidden),
            'd', self.n_hidden
        )
        return self.root.evaluate(U)
    #
    def gradient(self, X):
        grad = np.empty(self.n_param, 'd')

        U = np.fromiter(
            (mod.evaluate(X) for mod in self.hidden),
            'd', self.n_hidden
        )
        s = self.root.evaluate(U)

        grad[:self.root.n_param] = self.root.gradient(U)
        m = self.root.n_param
        grad_x_root = self.root.gradient_x(U)
        for i,mod in enumerate(self.hidden):
            grad[m:m+mod.n_param] = mod.gradient(X) * grad_x_root[i]
            m += mod.n_param
        return grad

class MLPerceptron1(Model):
    #
    # X1 (N, n+1)
    # W (m, n+1)
    def evaluate(self, X):
        func = self.func
        
        N = len(X)
        X1 = np.empty((N, self.n_input+1), np.double)
        X1[:,0] = 1
        X1[:,1:] = X

        U = X @ self.W.T
        for j in range(N):
            U[:] = func.evaluate(U[j])