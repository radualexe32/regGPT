import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
class LogisticRegression:
    def __init__(self, rate = 0.001, epochs = 1000):
        self.rate = rate
        self.epochs = epochs 
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0 

        for _ in range(self.epochs):
            pred = np.dot(X, self.w) + self.b
            sig_pred = sigmoid(pred)

            dw = (1 / samples) * np.dot(X.T, (sig_pred - y))
            db = (1 / samples) * np.sum(sig_pred - y)

            self.w -= self.rate * dw
            self.b -= self.rate * db

    def predict(self, X):
        pred = np.dot(X, self.w) + self.b
        y_hat = sigmoid(pred)
        return [0 if i < 0.5 else 1 for i in y_hat]