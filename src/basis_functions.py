import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from train import MSE
from slr import LinearRegression

X, y = make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 42)
y = y**2  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
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

    plt.scatter(X_train, y_train)
    plt.plot(X_plot, y_plot, color='red')
    plt.show()
