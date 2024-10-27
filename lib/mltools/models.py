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
    def init_param(self, param):
        if self.param is None:
            self.param = param
        else:
            self.param[:] = param
    #
    def evaluate_one(self, Xk):
        raise NotImplementedError()
    #
    def gradient_one(self, Xk):
        raise NotImplementedError()
    #
    def gradient_x_one(self, Xk):
        raise NotImplementedError()
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
    def igradient_x(self, X):
        gradient_x_one = self.gradient_x_one
        for Xk in X:
            yield gradient_x_one(Xk)
    #
    def gradient(self, X):
        rows = tuple(self.igradient(X))
        return np.vstack(rows)
    #
    def gradient_x(self, X):
        rows = tuple(self.igradient_x(X))
        return np.vstack(rows)
        
class LinearModel(Model):
    #
    def __init__(self, n, param=None):
        self.n_input = n
        self.n_param = n + 1
        self.param = None
        if param is None:
            self.param = None
        else:
            self.init_param(param)
    #
    def evaluate_one(self, Xk):
        return self.param[0] + (self.param[1:] @ Xk)
    #
    def evaluate(self, X):
        return self.param[0] + X @ self.param[1:]
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
    def gradient_x(self, X):
        return self.param[None,1:].repeat(X.shape[0], axis=0)

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
        Y = self.param[0] + X @ self.param[1:]
        return self.outfunc.evaluate(Y)
    #
    def gradient(self, X):
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
    def evaluate(self, X):
        N = X.shape[0]
        S = np.empty((N, self.n_hidden), np.double)
        for j, mod in enumerate(self.hidden):
            S[:,j] = X @ mod.param[1:] + mod.param[0] # mod.evaluate(X)
        U = self.outfunc.evaluate(S)
        return self.root.evaluate(U)
    #
    def gradient(self, X):
        N = X.shape[0]

        S = np.empty((N, self.n_hidden), np.double)
        Grad = np.empty((N,self.n_param), np.double)

        X1 = np.empty((N, self.n_input+1), np.double)
        X1[:,0] = 1
        X1[:,1:] = X
        
        for j, mod in enumerate(self.hidden):
            S[:,j] =  X1 @ mod.param # mod.evaluate(X)

        U = self.outfunc.evaluate(S)
        Grad[:,:self.root.n_param] = self.root.gradient(U)

        Gx = self.root.param[1:] # (N, m+1)
        
        D = self.outfunc.derivative(S)
    
        m = self.root.n_param
        for j, mod in enumerate(self.hidden):
            G = X1 * (D[:,j:j+1] * Gx[j]) #DG[:,j:j+1]
            Grad[:,m:m+mod.n_param] = G
            m += mod.n_param
        return Grad

class LinearLayer:

    def __init__(self, n_input, n_output, param=None):
        self.n_input  = n_input
        self.n_output = n_output
        self.n_param  = n_output * (n_input+1)
        self.models   = []
        self.param    = None
        if param is not None:
            self.init_param(param)
    #
    def init_param(self, param):
        if param.shape[0] != self.n_param:
            raise TypeError("param.shape[0] != self.n_param")
        self.param = param
        n_param = self.n_input+1
        for j in range(self.n_output):
            mod = LinearModel(self.n_input, param=self.param[start:start+n_param])
            self.models.append(mod)
            start += n_param
    #
    def forward(self, X):
        N = X.shape[0]
        Y = np.empty((N, self.n_output), np.double)
        
        for j in range(self.n_output):
            mod = self.models[j]
            Y[:,j] = mod.evaluate(X)
            
        return Y
    #
    def backward(self, X, D):
        N = X.shape[0]
        G = np.empty((N, self.n_param), "d")
        Gx = np.zeros((N, self.n_input), "d")
        mod_n_param = self.n_param
        start = 0
        for j in range(self.n_output):
            mod = self.models[j]
            G[:,start:start+mod_n_param] = mod.gradient(X) * D[j]
            start += mod_n_param
            Gx += mod.gradient_x(X) * D[j]
        return G, Gx
    #
