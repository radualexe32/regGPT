import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from slr import LinearRegression
from lor import LogisticRegression

def MSE(y_test, pred):
    return np.mean((y_test - pred) ** 2)

def plot(X, X_train, y_train, X_test, y_test, y_pred_line):
    plt.figure(figsize = (8, 6))
    plt.scatter(X_train, y_train)
    plt.plot(X, y_pred_line, color = "red", linewidth = 1, label = "Prediction")
    plt.show()

def train_slr():
    X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)
    reg = LinearRegression(rate = 0.01, epochs = 1000)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    print(f"MSE = {MSE(y_test, pred)}")
    print(f"Reg Line: y = {reg.w[0]}x + {reg.b}")

    fig, ax = plt.subplots()
    line, = ax.plot(X, reg.w_hist[0] * X + reg.b_hist[0], 'r')  
    plt.scatter(X_train, y_train)
    textbox = ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top')

    def animate(i):
        line.set_ydata(reg.w_hist[i] * X + reg.b_hist[i]) 
        textbox.set_text(f'w = {reg.w_hist[i][0]:.2f}\nb = {reg.b_hist[i]:.2f}') 
        return line, textbox,

    ani = animation.FuncAnimation(fig, animate, frames=range(len(reg.w_hist)), interval=5, repeat=False) 
    plt.show()
    
def train_lor():
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

    reg = LogisticRegression(rate = 0.01)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    print(f"Accuracy = {np.sum(pred == y_test) / len(y_test)}")