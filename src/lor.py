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
            pred = sigmoid(np.dot(X, self.w) + self.b)

            self.w -= self.rate * (1 / samples) * np.dot(X.T, (pred - y))
            self.b -= self.rate * np.sum(pred - y)

    def predict(self, X):
        y_hat = sigmoid(np.dot(X, self.w) + self.b)
        return [0 if i < 0.5 else 1 for i in y_hat]