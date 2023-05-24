import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from slr import LinearRegression
from lor import LogisticRegression

def MSE(y_test, pred):
    return np.mean((y_test - pred) ** 2)

def plot(X, X_train, y_train, X_test, y_test, y_pred_line):
    cmap = plt.get_cmap("viridis")
    plt.figure(figsize = (8, 6))
    plt.scatter(X_train, y_train, color = cmap(0.9), s = 10)
    plt.scatter(X_test, y_test, color = cmap(0.5), s = 10)
    plt.plot(X, y_pred_line, color = "black", linewidth = 1, label = "Prediction")
    plt.show()

def train_slr():
    X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 10, random_state = 3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    print(f"MSE = {MSE(y_test, pred)}")
    print(f"Reg Line: y = {reg.w[0]}x + {reg.b}")

    y_pred_line = reg.predict(X)
    plot(X, X_train, y_train, X_test, y_test, y_pred_line)

def train_lor():
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

    reg = LogisticRegression(rate = 0.01)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    print(f"Accuracy = {np.sum(pred == y_test) / len(y_test)}")