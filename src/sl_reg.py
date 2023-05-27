import numpy as np

def MSE(y_test, pred):
    return np.mean((y_test - pred) ** 2)

def R2(y_test, pred):
    return 1 - (np.sum((y_test - pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

class LinearRegression:
    def __init__(self, rate = 0.001, epochs = 1000):
        self.rate = rate
        self.epochs = epochs
        self.w, self.b = None, None
        self.w_hist, self.b_hist, self.mse_hist, self.r2_hist = [[] for _ in range(4)]

    def fit(self, X, y):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.epochs):
            y_hat = np.dot(X, self.w) + self.b
            self.w -= self.rate * (2 / samples) * np.dot(X.T, (y_hat - y))
            self.b -= self.rate * (2 / samples) * np.sum(y_hat - y)
            self.w_hist.append(self.w.copy())
            self.b_hist.append(self.b)
            self.mse_hist.append(MSE(y, y_hat))
            self.r2_hist.append(R2(y, y_hat))

    def predict(self, X):
        return np.dot(X, self.w) + self.b