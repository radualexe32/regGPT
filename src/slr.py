import numpy as np

class LinearRegression:
    def __init__(self, rate = 0.001, epochs = 1000):
        self.rate = rate
        self.epochs = epochs
        self.w = None 
        self.b = None
        self.w_hist, self.b_hist = [], []

    def fit(self, X, y):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.epochs):
            y_hat = np.dot(X, self.w) + self.b
            self.w -= self.rate * (1 / samples) * np.dot(X.T, (y_hat - y))
            self.b -= self.rate * (1 / samples) * np.sum(y_hat - y)
            self.w_hist.append(self.w.copy())
            self.b_hist.append(self.b)

    def predict(self, X):
        return np.dot(X, self.w) + self.b