import numpy as np
class LinearModel():
    def __init__(self):
        self.X = None
        self.y = None
        self.coefs = None
        self.sse = None
        self.sst = None
    def compute_cost(self, X, y, theta):
        predictions = X.dot(theta)
        sse = np.sum(np.square(y - predictions))
        sst = np.sum(np.square(y - np.mean(y)))
        return sse, sst
    def regression(self, X, y):
        n = y.size
        see = 0
        theta = np.dot(np.linalg.pinv(X), y)
        see, sst = self.compute_cost(X, y, theta)
        return np.around(theta.flatten(), 3), see, sst
    def train(self, X, y):
        self.X = X        
        self.y = y
        m = len(self.X[0]) + 1
        it = np.ones(shape=(y.size, m))
        it[:, 1:m] = self.X
        self.coefs, self.sse, self.sst = self.regression(it, self.y) 
        return self
    def predict(self, X):
        if X.ndim == 1:
            m = len(X) + 1
            it = np.ones(shape=(1, m))
        else: 
            m = len(X[0]) + 1
            it = np.ones(shape=(len(X), m))
        it[:, 1:m] = X
        pred = it.dot(self.coefs)
        return pred
    def r2(self, X, y):
        m = len(X[0]) + 1
        it = np.ones(shape=(y.size, m))
        it[:, 1:m] = X
        self.train(X, y)
        result = 1 - (self.sse / self.sst)
        return result