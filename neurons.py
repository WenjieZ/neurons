import numpy as np

class Neurons:
    def __init__(self, W = None, b = None, f = None, df = None):
        self.W_ = W
        self.b_ = b
        self.f_ = f
        self.df_ = df
        
    def __gd__(self, X, Y, stepSize = 1):
        N = X.shape[0]
        if N == 0:
            return
        
        XW = np.dot(X, self.W_)
        f, df = self.f_(XW), self.df_(XW)
        B = np.tile(self.b_, (N, 1))
        
        self.W_ -= stepSize / N * np.dot(X.T, df*(f+B-Y))
        if self.f_.__name__ == 'relu':
            self.b_ -= stepSize * (np.mean(f - Y, axis=0) + self.b_)
        else:
            self.b_ = np.mean(Y - f, axis=0)
    
    def predict(self, X):
        N = X.shape[0]
        return self.f_(np.dot(X, self.W_)) + np.tile(self.b_, (N, 1))
    
    def score(self, X, Y):
        N = X.shape[0]
        Y2 = self.predict(X)
        return np.sum((Y - Y2)**2) / N
        
    def rSquared(self, X, Y):
        m = Y.shape[1]
        Y2 = self.predict(X)
        r = 0
        for i in range(m):
            y1 = Y[:, i]
            y2 = Y2[:, i]
            r += np.sum((y1-y2)**2) / np.sum((y1-np.mean(y1))**2)
        r /= m
        return r
    
    def fit(self, X, Y, maxIter = 10, threshold = 1e-5, batchSize = 100, stepSize = 1, verbose = False, **kwargs):
        ## init
        N, n= X.shape
        m = Y.shape[1]
        if self.W_ is None:
            self.W_ = np.zeros((n, m))
        if self.b_ is None:
            self.b_ = np.zeros(m)
        
        ## optimization
        old_score = self.score(X[0:min(1000, N), :], Y[0:min(1000, N), :])
        if verbose:
            print("Score: ", old_score)
        batchs = N // batchSize
        scale = np.mean(np.abs(Y))
        
        for i in range(maxIter):
            for j in range(batchs):
                s1 = j * batchSize
                s2 = s1 + batchSize
                self.__gd__(X[s1:s2, :], Y[s1:s2, :], stepSize)
            self.__gd__(X[s2:, :], Y[s2:, :], stepSize)
                
            new_score = self.score(X[0:min(1000, N), :], Y[0:min(1000, N), :])
            if verbose:
                print("Score: ", new_score)
            if (old_score - new_score) / scale < threshold:
                return "Reached the threshold."
            else:
                old_score = new_score
        
        return "Reached the maximal iteration number."
