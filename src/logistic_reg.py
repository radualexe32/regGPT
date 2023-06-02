import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
class LogisticRegression:
    def __init__(self, rate = 0.001, epochs = 1000):
        self.rate = rate
        self.epochs = epochs 
        self.w, self.b = None, None
    
    def fit(self, X, y):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0 

        for _ in range(self.epochs):
            pred = sigmoid(np.dot(X, self.w) + self.b)
            self.w -= self.rate * (2 / samples) * np.dot(X.T, (pred - y))
            self.b -= self.rate * (2/ samples) * np.sum(pred - y)
        
    def fit_mini_batch(self, X, y, batch_size = 32):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0 

        for _ in range(self.epochs):
            idx = np.random.permutation(samples)
            X, y = X[idx], y[idx]
            
            for i in range(0, samples, batch_size):
                X_i = X[i:i + batch_size]
                y_i = y[i:i + batch_size]
                y_hat = sigmoid(np.dot(X_i, self.w) + self.b)
                self.w -= self.rate * (2 / samples) * np.dot(X_i.T, (y_hat - y_i))
                self.b -= self.rate * (2/ samples) * np.sum(y_hat - y_i)

    def predict(self, X):
        return [0 if i < 0.5 else 1 for i in sigmoid(np.dot(X, self.w) + self.b)]