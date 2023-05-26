import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from train import MSE
from slr import LinearRegression
from train import *

plt.style.use("dark_background")

X, y, X_train, X_test, y_train, y_test = data(degree = 2)
    
def basis_train(reg = LinearRegression(), degrees = 2):
    model = make_pipeline(PolynomialFeatures(degrees), reg)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    print(f"MSE = {MSE(y_test, y_hat)}")
    return model
    
if __name__ == "__main__":
    model = basis_train()
    X_plot = np.linspace(X.min(), X.max(), 100)[:, None]
    y_plot = model.predict(X_plot)
    
    plt.scatter(X_train, y_train, c = "peru")
    plt.plot(X_plot, y_plot, "aqua", lw = 5)
    plt.plot(X_plot, y_plot, "white", lw = 2)
    plt.show()