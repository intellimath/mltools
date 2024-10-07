import numpy as np
# from mltools.aggfuncs import ArithMean
from math import sqrt

np_zeros = np.zeros
np_zeros_like = np.zeros_like
np_mean = np.mean

class Risk:
    #
    def __init__(self, model, loss_func): 
        self.model = model
        self.loss_func = loss_func
    #
    def evaluate_losses(self, X, Y):
        YY = self.model.evaluate(X)
        L = self.loss_func.evaluate(YY, Y)
        return L
    #
    def evaluate(self, X, Y):
        YY = self.model.evaluate(X)
        L = self.loss_func.evaluate(YY, Y)
        return L.mean()
    #
    def gradient(self, X, Y):
        YY = self.model.evaluate(X)
        V = self.loss_func.derivative(YY, Y)
        G = self.model.gradient(X)
        GV = G * V[:,None]
        return GV.mean(axis=0)

class WRisk:
    #
    def __init__(self, model, loss_func, weights=None): 
        self.model = model
        self.loss_func = loss_func
        self.weights = None
    #
    def evaluate_losses(self, X, Y):
        YY = self.model.evaluate(X)
        L = self.loss_func.evaluate(YY, Y)
        return L
    #
    def evaluate(self, X, Y):
        if self.weights is None:
            N = X.shape[0]
            self.weights = np.full(N, 1.0/N, 'd')
        YY = self.model.evaluate(X)
        L = self.loss_func.evaluate(YY, Y)
        return np.average(L, weights=self.weights)
    #
    def gradient(self, X, Y):
        YY = self.model.evaluate(X)
        V = self.loss_func.derivative(YY, Y) * self.weights
        gradient = self.model.gradient_one
        return sum([vk * gradient(Xk) for vk, Xk in zip(V, X)])

# class Risk2:
#     #
#     def __init__(self, model, loss_func, agg=None):
#         self.model = model
#         self.loss_func = loss_func
#         if agg is None:
#             self.agg = ArithMean()
#         else:
#             self.agg = agg
#     #
#     def evaluate(self, X, Y):
#         model_evaluate = self.model.evaluate
#         YY = np.fromiter(
#                 (model_evaluate(Xk) for Xk in X), 
#                 'd', len(X))
#         L = self.loss_func.evaluate(YY, Y)
#         return self.agg.evaluate(L)
#     #
#     def gradient(self, X, Y):
#         model_evaluate = self.model.evaluate
#         YY = np.fromiter(
#                 (model_evaluate(Xk) for Xk in X), 
#                 'd', len(X))
#         L = self.loss_func.evaluate(YY, Y)
#         G = self.agg.gradient(L)
#         V = G * self.loss_func.derivative(YY, Y)
#         # grad = np_zeros(self.model.n_param, 'd')
#         gradient = self.model.gradient
#         # for vk, Xk in zip(V, X):
#         #     grad += vk * gradient(Xk)
#         return sum([vk * gradient(Xk) for vk, Xk in zip(V, X)])
#         # return grad


