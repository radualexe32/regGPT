import numpy as np
import time 

def MSE(y_test, pred):
    return np.mean((y_test - pred) ** 2)

def R2(y_test, pred):
    return 1 - (np.sum((y_test - pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

class LinearRegression:
    def __init__(self, rate = 0.001, epochs = 1000):
        self.rate = rate
        self.epochs = epochs
        self.w, self.b = None, None
        self.w_hist, self.b_hist, self.mse_hist, self.r2_hist, self.time_hist = [[] for _ in range(5)]

    def fit(self, X, y):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0
        start = time.time()
        
        for _ in range(self.epochs):
            y_hat = np.dot(X, self.w) + self.b
            self.w -= self.rate * (2 / samples) * np.dot(X.T, (y_hat - y))
            self.b -= self.rate * (2 / samples) * np.sum(y_hat - y)
            self.w_hist.append(self.w.copy())
            self.b_hist.append(self.b)
            self.mse_hist.append(MSE(y, y_hat))
            self.r2_hist.append(R2(y, y_hat))
            self.time_hist.append(time.time() - start)

    def fit_mini_batch(self, X, y, batch_size = 32, tolerance = 1e-5):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0
        start = time.time()
    
        for _ in range(self.epochs):
            idx = np.random.permutation(samples)
            X, y = X[idx], y[idx]

            mse_prev = np.inf
            for i in range(batch_size):
                X_i = X[i:i + batch_size]
                y_i = y[i:i + batch_size]
                y_hat = np.dot(X_i, self.w) + self.b
                self.w -= self.rate * (2 / samples) * np.dot(X_i.T, (y_hat - y_i))
                self.b -= self.rate * (2 / samples) * np.sum(y_hat - y_i)

                y_hat_all= np.dot(X, self.w) + self.b
                mse_curr = MSE(y, y_hat_all)
                self.w_hist.append(self.w.copy())
                self.b_hist.append(self.b)
                self.mse_hist.append(MSE(y, y_hat_all))
                self.r2_hist.append(R2(y, y_hat_all))
                self.time_hist.append(time.time() - start)

                if mse_prev - mse_curr < tolerance:
                    return 

                mse_prev = mse_curr

    def predict(self, X):
        return np.dot(X, self.w) + self.b