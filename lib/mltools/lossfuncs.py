import numpy as np

class Loss:
    #
    def evaluate(self, Y, Yp):
        pass
    #
    def derivative(self, Y, Yp):
        pass

class ErrorLoss(Loss):
    #
    def __init__(self, func):
        self.func = func
    #
    def evaluate(self, Y, Yp):
        return self.func.evaluate(Y - Yp)
    #
    def derivative(self, Y, Yp):
        return self.func.derivative(Y - Yp)

class MarginLoss(Loss):
    #
    def __init__(self, rho_func):
        self.rho_func = rho_func
    #
    def evaluate(self, Y, Yp):
        return self.rho_func.evaluate(Y*Yp)
    #
    def derivative(self, Y, Yp):
        return Yp * self.rho_func.derivative(Y*Yp)

