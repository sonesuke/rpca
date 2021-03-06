import numpy as np

def softThreshold1d(x, penalty):
    for i in range(x.shape[0]):
        x[i] = np.sign(x[i]) * np.max([np.abs(x[i]) - penalty, 0])
    return x


def softThreshold2d(x, penalty):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = np.sign(x[i, j]) * np.max([np.abs(x[i, j]) - penalty, 0])
    return x


def l1norm(x):
    l1norm = 0.0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            l1norm += np.abs(x[i, j])
    return l1norm


def norm2Squared(x):
    l1norm = 0.0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            l1norm += x[i, j] * x[i, j]
    return l1norm


class RPCA:

    MAX_ITERS = 228

    def __init__(self):
        pass

    def calculate(self, X, lpenalty, spenalty):
        self.X = X
        self.lpenalty = lpenalty
        self.spenalty = spenalty

        self.L = np.zeros(X.shape, dtype=float)
        self.S = np.zeros(X.shape, dtype=float)
        self.E = np.zeros(X.shape, dtype=float)

        return self.computeRSVD()


    def computeRSVD(self):
        mu = self.X.shape[0] * self.X.shape[1] / (4 * l1norm(X))
        objPrev = 0.5 * norm2Squared(X)
        obj = objPrev
        tol = 1e-8 * objPrev
        diff = 2 * tol

        it = 0
        while diff > tol and it < MAX_ITERS:
            nuclear_norm = self.computeS(mu)
            l1Norm = self.computeL(mu)
            l2Norm = self.computeE()
            obj = self.computeObjective(nuclear_norm, l1Norm, l2Norm)
            diff = np.abs(objPrev - obj)
            objPrev = obj
            mu = self.computeDynamicMu()
            it += 1

        return self.L, self.S, self.E

    def computeL(self, mu):
        LPenalty = self.lpenalty * mu
        u, d, v = np.linalg.svd(self.X - self.S, full_matrices=False)
        penalizedD = softThreshold1d(d, LPenalty)
        D = np.diag(penalizedD)
        self.L = np.dot(u, np.dot(D, v))
        return np.sum(penalizedD) * LPenalty

    def computeS(self, mu):
        SPenalty = self.spenalty * mu
        penalizedS = softThreshold2d(self.X - self.L, SPenalty)
        self.S = penalizedS
        return l1norm(penalizedS) * SPenalty

    def computeE(self):
        self.E = self.X - self.L - self.S
        return norm2Squared(self.E)

    def computeObjective(self, nuclaernorm, l1norm, l2norm):
        return 0.5 * l2norm + nuclaernorm + l1norm

    def computeDynamicMu(self):
        m = self.E.shape[0]
        n = self.E.shape[1]

        E_sd = np.std(np.reshape(self.E, (m * n, 1)))
        mu = E_sd * np.sqrt(2 * np.max((m, n)))
        return np.max((0.01, mu))


import csv

ts = []
with open("outlier_ts_periodic.csv", "r") as f:
    r = csv.reader(f)
    r.next()
    for row in r:
        ts.append(float(row[1]))

ts = np.array(ts,  dtype=float)

lpenalty = 0.0
spenalty = 0.0
MAX_ITERS = 500

X = np.zeros((14, 24), dtype=float)
a = 0
x = 0
while a < ts.shape[0]:
    y = 0
    while y < 24:
        X[x, y] = ts[a]
        y += 1
        a += 1
    x += 1

X = np.transpose(X)

rpca = RPCA()
#L, S, E = rpca.calculate(X, 1.0, 0.2857738033247)
L, S, E = rpca.calculate(X, 1.0, 0.2)

import matplotlib.pyplot as plt

xp_s = np.reshape(np.transpose(L), (ts.shape[0], 1))
ep_s = np.reshape(np.transpose(E), (ts.shape[0], 1))
sp_s = np.reshape(np.transpose(S), (ts.shape[0], 1))


print ts.shape
print xp_s.shape
plt.plot(ts)
plt.plot(xp_s)
plt.plot(ep_s)
plt.plot(sp_s)
plt.show()
