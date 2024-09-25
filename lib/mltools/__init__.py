#

def regression(X, Y, model, loss_func, h=0.01, tol=1.0e-8, n_iter=1000):
    risk = risks.Risk(model, loss_func)
    gd = gda.GradientDescent(risk, h=h, tol=tol, n_iter=n_iter)
    gd.fit(X, Y)
    return gd

