import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from math import sqrt
import sys

import mltools.aggfuncs as aggfuncs

from sklearn.cluster import kmeans_plusplus

def norm2(x):
    return (x @ x)

def mnorm2(x, S1):
    return (S1 @ x) @ x

# def euclid_norm2(X, c):
#     """
#     Function return `Xc @ Xc.T`, where `Xc = X - c`.
#     """
#     delta = X - c
#     return np.einsum('nj,nj->n', delta, delta, optimize=True)

# def mahalanobis_norm2(X, c, S1):
#     """
#     Function return `((Xc @ S1)*Xc).sum()`, where `Xc = X - c`.
#     """
#     delta = X - c
#     return np.einsum('nj,jk,nk->n', delta, S1, delta, optimize=True)

class KMeans:
    #
    def __init__(self, q, tol=1.0e-9, n_iter=1000):
        self.q = q
        self.n_iter = n_iter
        self.tol = tol
    #
    def predict(self, X):
        Y = np.empty(len(X), 'i')
        Is = self.find_clusters(X)
        for j in range(self.q):
            Ij = Is[j]
            Y[Ij] = j
        return Y
    #
    def eval_dists(self, X):
        Ds = []
        for x in X:
            ds = [norm2(x - cj) for cj in self.c]
            dmin = min(ds)
            Ds.append(dmin)
        return np.sqrt(np.array(Ds))
    #
    def dist(self, x):
        ds = [norm2(x - cj) for cj in self.c]
        dmin = min(ds)
        return sqrt(dmin)
    #
    def find_clusters(self, X):
        array = np.array
        Is = [[] for j in range(self.q)]
        qval = 0
        for k, xk in enumerate(X):
            ds = [norm2(xk - cj) for cj in self.c]
            dmin = min(ds)
            qval += dmin
            for j, dj in enumerate(ds):
                if dj == dmin:
                    Is[j].append(k)
        Is = [array(Ij) for Ij in Is]
        self.qvals.append(sqrt(qval))
        return Is
    #
    def find_locations(self, X, Is):
        c = np.empty((self.q, X.shape[1]), 'd')
        for j in range(self.q):
            Ij = Is[j]
            Xj = X[Ij]
            c[j,:] = Xj.mean(axis=0)
        return c
    #
    def stop_condition(self, c, c_prev):
        dc = c - c_prev
        dc2 = dc * dc
        dmax2 = max(dc2.sum(axis=1))
        dmax = sqrt(dmax2)
        print(dmax)
        self.dvals.append(dmax)
        if dmax < self.tol:
            return True
        
        return False
    #
    def initial_locations(self, X):
        return kmeans_plusplus(X, self.q)[0]
        # N, n = X.shape
        # q = self.q
        
        # xmin = X.min(axis=0)
        # xmax = X.max(axis=0)
        
        # c = np.random.random(size=(q, n))
        # return xmin + c * (xmax - xmin)
    #
    def fit(self, X):
        self.c = self.initial_locations(X)
        self.dvals = []
        self.qvals = []
        for K in range(self.n_iter):
            c_prev = self.c.copy()
            self.Is = self.find_clusters(X)
            self.c = self.find_locations(X, self.Is)
            if self.stop_condition(self.c, c_prev):
                break
        self.K = K + 1

class KMeansMahalanobis:
    #
    def __init__(self, q, tol=1.0e-8, n_iter_c=100, n_iter_s=22, n_iter=500):
        self.q = q
        self.n_iter = n_iter
        self.n_iter_c = n_iter_c
        self.n_iter_s = n_iter_s
        self.tol = tol
        self.qvals = []
    #
    def predict(self, X):
        Y = np.empty(len(X), 'i')
        Is = self.find_clusters(X)
        for j in range(self.q):
            Ij = Is[j]
            Y[Ij] = j
        return Y
    #
    def dist(self, x):
        S = self.S
        c = self.c
        ds = [mnorm2(x - c[j], S[j]) for j in range(self.q)]
        d_min = min(ds)
        return sqrt(d_min)
    #
    def eval_dists(self, X):
        Ds = []
        S = self.S
        c = self.c
        for x in X:
            ds = [mnorm2(x - c[j], S[j]) for j in range(self.q)]
            dmin = min(ds)
            Ds.append(dmin)
        return np.sqrt(np.array(Ds))
    #
    def find_clusters(self, X):
        array = np.array
        q = self.q
        Is = [[] for j in range(q)]
        S = self.S
        c = self.c
        for k, xk in enumerate(X):
            ds = [mnorm2(xk - c[j], S[j]) for j in range(q)]
            d_min = min(ds)
            for j in range(q):
                if ds[j] == d_min:
                    Is[j].append(k)
        Is = [array(Ij) for Ij in Is]
        return Is
    #
    def eval_qval(self, X):
        q = self.q
        S = self.S
        c = self.c
        qval = 0
        for xk in X:
            qval += min([mnorm2(xk - c[j], S[j]) for j in range(q)])
        return np.sqrt(qval)
    #
    def find_locations(self, X, Is):
        mean = np.mean
        c = np.zeros((self.q, X.shape[1]), 'd')
        for j in range(self.q):
            Ij = Is[j]
            Xj = X[Ij]
            c[j,:] = Xj.mean(axis=0)
        return c
    #
    def initial_locations(self, X):
        return kmeans_plusplus(X, self.q)[0]
        # N, n = X.shape
        # q = self.q
        
        # xmin = X.min(axis=0)
        # xmax = X.max(axis=0)
        # c = np.random.random(size=(q, n))
        # return xmin + c * (xmax - xmin)
    #
    def find_covs(self, X, Is):
        outer = np.outer
        zeros = np.zeros
        inv = linalg.inv
        det = linalg.det
        c = self.c
        n = X.shape[1]
        S = []
        n1 = 1.0/n
        s = 0
        for j in range(self.q):
            # Sj = zeros((n,n), 'd')
            Ij = Is[j]
            Xc_j = X[Ij] - c[j]
            XXj = Xc_j.T @ Xc_j
            Lj = (XXj @ self.S[j]).sum()
            Sj1 = XXj / Lj
            Sj1 /= det(Sj1) ** n1 
            Sj = inv(Sj1)
            # s += det(Sj1) ** n1
            S.append(Sj)

        # Ds = [det(Sj) ** n1 for Sj in S1]
        # s = sum(Ds)
        # Ds = [d/s for d in Ds]

        # S1 = [Sj/dj for Sj, dj in zip(S1, Ds)]
        return S
    #
    def stop_condition(self, qval, qval_prev):
        if abs(qval - qval_prev) / (1 + self.qval_min) < self.tol:
            return True
        
        return False
    #
    def fit_locations(self, X):
        self.qval_min = qval = self.eval_qval(X)
        self.c_min = self.c.copy()
        for K in range(self.n_iter_c):
            qval_prev = qval
            self.Is = Is = self.find_clusters(X)
            self.c = self.find_locations(X, Is)

            qval = self.eval_qval(X)
            self.qvals.append(qval)
            if qval < self.qval_min:
                self.qval_min = qval
                self.c_min = self.c.copy()

            if self.stop_condition(qval, qval_prev):
                break

        self.c = self.c_min
    #
    def fit_scatters(self, X):
        self.qval_min = qval = self.eval_qval(X)
        self.S_min = [S.copy() for S in self.S]
        for K in range(self.n_iter_s):
            qval_prev = qval
            self.Is = Is = self.find_clusters(X)
            self.S = self.find_covs(X, Is)

            qval = self.eval_qval(X)
            self.qvals.append(qval)
            if qval < self.qval_min:
                self.qval_min = qval
                self.S_min = [S.copy() for S in self.S]

            if self.stop_condition(qval, qval_prev):
                break

        self.S = [S.copy() for S in self.S_min]
    #
    def fit(self, X):
        n = X.shape[1]
        self.c = self.c_min = self.initial_locations(X)
        self.S = self.S_min = [np.identity(n) for j in range(self.q)]
        self.qvals = []
        self.qvals2 = []
        qval2 = self.qval_min = self.eval_qval(X)
        for K in range(self.n_iter):
            qval_prev2 = qval2
            self.fit_locations(X)
            self.fit_scatters(X)
            qval2 = self.eval_qval(X)
            self.qvals2.append(qval2)
            if self.stop_condition(qval2, qval_prev2):
                break

        self.K = K + 1

class RKMeansMahalanobis:
    #
    def __init__(self, q, avrfunc=None, tol=1.0e-9, n_iter_c=44, n_iter_s=22, n_iter=500):
        self.q = q
        self.n_iter = n_iter
        self.n_iter_c = n_iter_c
        self.n_iter_s = n_iter_s
        self.tol = tol
        if avrfunc is None:
            self.avrfunc = aggfuncs.ArithMean()
        else:
            self.avrfunc = avrfunc
    #
    def norm2(self, x):
        S1 = self.S1
        c = self.c
        return min((mnorm2(x - c[j], S1[j]) for j in range(self.q)))
    #
    def find_clusters(self, X):
        q = self.q
        I = [[] for j in range(q)]
        S1 = self.S1
        c = self.c
        for k, xk in enumerate(X):
            ds = [mnorm2(xk - c[j], S1[j]) for j in range(q)]
            d_min = min(ds)
            for j in range(q):
                if ds[j] == d_min:
                    I[j].append(k)
        return I
    #
    def find_locations(self, X, Is, G):
        n = X.shape[1]
        zeros = np.zeros
        c = zeros((self.q, n), 'd')
        for j in range(self.q):
            Ij = Is[j]
            cj = sum((G[k] * X[k] for k in Ij), start=zeros(n, 'd'))
            GG = sum(G[k] for k in Ij)
            c[j,:] = cj / GG
        return c
    #
    def eval_dists(self, X):
        Ds = []
        S1 = self.S1
        c = self.c
        for x in X:
            ds = [mnorm2(x - c[j], S1[j]) for j in range(self.q)]
            dmin = min(ds)
            Ds.append(dmin)
        return np.sqrt(np.array(Ds))
    #
    def initial_locations(self, X):
        return kmeans_plusplus(X, self.q)[0]
        # N, n = X.shape
        # q = self.q
        
        # xmin = np.fromiter((min(X[:,i]) for i in range(n)), 'd', n)
        # xmax = np.fromiter((max(X[:,i]) for i in range(n)), 'd', n)
        # c = np.random.random(size=(q, n))
        # return xmin + c * (xmax - xmin)
    #
    def find_covs(self, X, I, G):
        n = X.shape[1]
        S1 = []
        n1 = 1.0/n
        for j in range(self.q):
            Sj = np.zeros((n,n), 'd')
            Ij = I[j]
            cj = self.c[j]
            g = 0
            for k in Ij:
                v = X[k] - cj
                Sj += G[k] * np.outer(v, v)
                g += G[k]
            Sj /= g
            Sj = linalg.pinv(Sj)
            Sj /= linalg.det(Sj) ** n1
            S1.append(Sj)
        return S1
    #
    def stop_condition(self, qval, qval_prev):
        if abs(qval - qval_prev) / (1 + self.qval_min) < self.tol:
            return True
        
        return False
    #
    def eval_qval(self, X):
        self.ds = np.fromiter((self.norm2(x) for x in X), 'd', len(X))
        dd = self.avrfunc.evaluate(self.ds)
        return np.sqrt(dd)
    #
    def fit_locations(self, X):
        N = X.shape[0]
        self.c_min = self.c.copy()

        self.ds = np.fromiter((self.norm2(x) for x in X), 'd', N)
        dd = self.avrfunc.evaluate(self.ds)
        qval = self.qval_min = np.sqrt(dd)
        for K in range(self.n_iter_c):
            qval_prev = qval
            self.ds = np.fromiter((self.norm2(x) for x in X), 'd', N)
            self.avrfunc.evaluate(self.ds)
            G = self.avrfunc.gradient(self.ds)
            self.Is = self.find_clusters(X)
            self.c = self.find_locations(X, self.Is, G)
            
            self.ds = np.fromiter((self.norm2(x) for x in X), 'd', N)
            dd = self.avrfunc.evaluate(self.ds)
            qval = np.sqrt(dd)
            self.qvals.append(qval)
            
            if qval < self.qval_min:
                self.qval_min = qval
                self.c_min = self.c.copy()
                
            if self.stop_condition(qval, qval_prev):
                break

        self.K = K + 1
        self.c = self.c_min        
    #
    def fit_scatters(self, X):
        N = X.shape[0]
        self.S1_min = [S1.copy() for S1 in self.S1]

        self.ds = np.fromiter((self.norm2(x) for x in X), 'd', N)
        dd = self.avrfunc.evaluate(self.ds)
        qval = self.qval_min = np.sqrt(dd)
        for K in range(self.n_iter_s):
            qval_prev = qval
            self.ds = np.fromiter((self.norm2(x) for x in X), 'd', N)
            self.avrfunc.evaluate(self.ds)
            G = self.avrfunc.gradient(self.ds)
            self.Is = self.find_clusters(X)
            self.S1 = self.find_covs(X, self.Is, G)
            
            self.ds = np.fromiter((self.norm2(x) for x in X), 'd', N)
            dd = self.avrfunc.evaluate(self.ds)
            qval = np.sqrt(dd)
            self.qvals.append(qval)

            if qval < self.qval_min:
                self.qval_min = qval
                self.S1_min = [S1.copy() for S1 in self.S1]
            
            if self.stop_condition(qval, qval_prev):
                break

        self.K = K + 1
        self.S1 = [S1.copy() for S1 in self.S1_min]
    #
    def fit(self, X):
        q = self.q
        n = X.shape[1]
        N = X.shape[0]
        self.c = self.c_min = self.initial_locations(X)
        self.S1 = self.S1_min = [np.identity(n) for j in range(q)]
        self.qvals = []
        self.qvals2 = []
        qval2 = self.qval_min = self.eval_qval(X)
        for K in range(self.n_iter):
            qval_prev2 = qval2
            self.fit_locations(X)
            self.fit_scatters(X)
            qval2 = self.eval_qval(X)
            self.qvals2.append(qval2)
            if self.stop_condition(qval2, qval_prev2):
                break
        self.K = K + 1
