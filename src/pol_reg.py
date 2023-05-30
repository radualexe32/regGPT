import numpy as np
from sl_reg import MSE, R2

class PolynomialRegression:
    def __init__(self, degree, rate = 0.001, epochs = 1000):
        self.degree = degree
        self.rate = rate
        self.epochs = epochs
        self.w, self.b = None, None
        self.mse_hist, self.r2_hist = [], []

    def fit(self, X, y):
        X = self.polynomial_features(X, self.degree)
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.epochs):
            y_hat = np.dot(X, self.w) + self.b
            self.w -= self.rate * (2 / samples) * np.dot(X.T, (y_hat - y))
            self.b -= self.rate * (2 / samples) * np.sum(y_hat - y)
            self.mse_hist.append(MSE(y, y_hat))
            self.r2_hist.append(R2(y, y_hat))

    def predict(self, X):
        X = self.polynomial_features(X, self.degree)
        return np.dot(X, self.w) + self.b

    @staticmethod
    def polynomial_features(X, degree):
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree)
        return poly.fit_transform(X)
