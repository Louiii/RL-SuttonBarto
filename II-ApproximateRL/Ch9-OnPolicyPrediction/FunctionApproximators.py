import numpy as np

class LinearApprox():
    def __init__(self):
        self.ω = np.array([0, 0])

    def x(self, s):# feature vector
        return np.array(s)

    def V(self, s):
        X = x(s)
        return np.dot(self.w, X)

    def gradient(self, s):
        return x(s)

