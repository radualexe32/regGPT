import numpy as np

class LinearRegression:
    def __init__(self, rate = 0.001, epochs = 1000):
        self.rate = rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        y_hat = np.dot(X, self.w) + self.b

        for _ in range(self.epochs):
            dw = (1 / samples) * np.dot(X.T, (y_hat - y))
            db = (1 / samples) * np.sum(y_hat - y)

            self.w -= self.rate * dw
            self.b -= self.rate * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b
