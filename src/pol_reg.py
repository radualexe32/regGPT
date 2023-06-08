import numpy as np
from sl_reg import MSE, R2

class PolynomialRegression:
    def __init__(self, degree, rate = 0.001, epochs = 1000):
        self.degree = degree
        self.rate = rate
        self.epochs = epochs
        self.w, self.b = None, None
        self.w_hist, self.b_hist, self.mse_hist, self.r2_hist = [[] for _ in range(4)]

    def fit(self, X, y):
        X = self.polynomial_features(X, self.degree)
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
        
    def fit_mini_batch(self, X, y, batch_size = 32, tolerance = 1e-5):
        X = self.polynomial_features(X, self.degree)
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.epochs):
            idx = np.random.permutation(samples)
            X, y = X[idx], y[idx]

            mse_prev = np.inf
            for i in range(0, samples, batch_size):
                X_i = X[i:i + batch_size]
                y_i = y[i:i + batch_size]
                y_hat = np.dot(X_i, self.w) + self.b
                self.w -= self.rate * (2 / samples) * np.dot(X_i.T, (y_hat - y_i))
                self.b -= self.rate * (2 / samples) * np.sum(y_hat - y_i)

                y_hat_all = np.dot(X, self.w) + self.b
                mse_curr = MSE(y, y_hat_all)
                self.w_hist.append(self.w.copy())
                self.b_hist.append(self.b)
                self.mse_hist.append(MSE(y, y_hat_all))
                self.r2_hist.append(R2(y, y_hat_all))

                if np.abs(mse_prev - mse_curr) < tolerance:
                    return

                mse_prev = mse_curr

    def predict(self, X):
        X = self.polynomial_features(X, self.degree)
        return np.dot(X, self.w) + self.b

    @staticmethod
    def polynomial_features(X, degree):
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree)
        return poly.fit_transform(X)
