import numpy as np
from scipy.special import expit

class Func:
    #
    def evaluate(self, X):
        """
        Вычисляет значения функции на массиве значений.
        X: массив значений
        Возвращает: массив значений фунции
        """
        raise NotImplementedError()
    #
    def derivative(self, X):
        """
        Вычисляет значения производной функции f'(x) на массиве значений.
        X: массив значений
        Возвращает: массив значений производной фунции
        """
        raise NotImplementedError()
    #
    def derivative2(self, X):
        """
        Вычисляет значения 2-й производной функции f''(x) на массиве значений.
        X: массив значений
        Возвращает: массив значений 2-й производной фунции
        """
        raise NotImplementedError()
    #
    def derivative2_scaled(self, X):
        """
        Вычисляет значения 2-й производной функции f''(x) на массиве значений (нормированный).
        X: массив значений
        Возвращает: массив значений 2-й производной фунции
        """
        return self.derivative2(X)
    #
    def derivative_div_x(self, X):
        """
        Вычисляет значения производной функции f'(x)/x на массиве значений.
        X: массив значений
        Возвращает: массив значений 2-й производной фунции
        """
        return self.derivative(X) / X
    
np_ones_like = np.ones_like
np_empty_like = np.empty_like
np_zeros_like = np.zeros_like
    
np_putmask = np.putmask
    
class Hinge(Func):
    #
    def __init__(self, delta=1.0):
        self.delta = delta
    #
    def evaluate(self, X):
        R = self.delta - X
        np.putmask(R, X > self.delta, 0)
        return R
    #
    def derivative(self, X):
        R = -np.ones_like(X)
        np.putmask(R, X > self.delta, 0)
        return R        
    #

class Square(Func):
    #
    def evaluate(self, X):
        return 0.5 * X * X
    #
    def derivative(self, X):
        return X
    #
    def derivative2(self, X):
        return np.ones_like(X)
    #
    def derivative_div_x(self, X):
        return np.ones_like(X)
    #

np_abs = np.abs
np_sign = np.sign
    
class Abs(Func):
    #
    def evaluate(self, X):
        return np.abs(X)
    #
    def derivative(self, X):
        return np.sign(X)
    #
    def derivative2(self, X):
        pass
    #
    def derivative_div_x(self, X):
        return np.sign(X) / X

np_sqrt = np.sqrt 

class SoftAbs(Func):
    #
    def __init__(self, eps=1.0):
        self.eps = eps
        self.eps2 = eps * eps
    #
    def evaluate(self, X):
        return np.sqrt(self.eps2 + X*X) - self.eps
    #
    def derivative(self, X):
        return X / np.sqrt(self.eps2 + X*X)
    #
    def derivative2(self, X):
        v = self.eps2 + X*X
        return self.eps2 / (v * np.sqrt(v))
    #
    def derivative2_scaled(self, X):
        v = self.eps2 + X*X
        return 1 / (v * np.sqrt(v))
    #
    def derivative_div_x(self, X):
        return 1 / np_sqrt(self.eps2 + X*X)

class SoftQuantileFunc(Func):
    #
    def __init__(self, func, alpha=0.5):
        self.rho = func
        self.alpha = alpha
    #
    def evaluate(self, X):
        Y = self.rho.evaluate(X)
        Y[X > 0] *= self.alpha
        Y[X < 0] *= (1.0 - self.alpha)
        return Y
    #
    def derivative(self, X):
        Y = self.rho.derivative(X)
        Y[X > 0] *= self.alpha
        Y[X < 0] *= (1.0 - self.alpha)
        return Y
    #
    def derivative2(self, X):
        Y = self.rho.derivative2(X)
        Y[X > 0] *= self.alpha
        Y[X < 0] *= (1.0 - self.alpha)
        return Y
    #
    def derivative2_scaled(self, X):
        Y = self.rho.derivative2_scaled(X)
        Y[X > 0] *= self.alpha
        Y[X < 0] *= (1.0 - self.alpha)
        return Y
    #
    def derivative_div_x(self, X):
        Y = self.rho.derivative_div_x(X)
        Y[X > 0] *= self.alpha
        Y[X < 0] *= (1.0 - self.alpha)
        return Y
    
np_log = np.log
np_exp = np.exp

class SoftHinge(Func):
    #
    def __init__(self, alpha = 1):
        self.alpha = alpha
    #
    def evaluate(self, X):
        return np.log(1.0 + np.exp(-self.alpha * X))
    #
    def derivative(self, X):
        v = np.exp(-self.alpha * X)
        return -self.alpha * v / (1 + v)
    #

class Logistic(Func):
    #
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    #
    def evaluate(self, X):
        return expit(self.alpha * X)
    #
    def derivative(self, X):
        V = expit(self.alpha * X)
        return self.alpha * V *  (1 - V)
    
class Sigmoidal(Func):
    #
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    #
    def evaluate(self, X):
        return np.tanh(self.alpha * X)
    #
    def derivative(self, X):
        V = np.cosh(self.alpha * X)
        return self.alpha / (V*V)

class RELU(Func):
    def evaluate(self, X):
        Y = X.copy()
        Y[X<0] = 0
        return Y
    #
    def derivative(self, X):
        Y = np.ones_like(X)
        Y[X<=0] = 0
        return Y

class ID(Func):
    #
    def evaluate(self, X):
        return X
    #
    def derivative(self, X):
        return np.ones_like(X)