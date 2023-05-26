from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from slr import LinearRegression

plt.style.use("seaborn-darkgrid")

def data(samples = 100, features = 1, noise = 20, degree = 1):
    X, y = datasets.make_regression(n_samples = samples, n_features = features, noise = noise, random_state = 42)
    y = y**degree
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)
    return X, y, X_train, X_test, y_train, y_test

def train_slr(rate = 0.01, epochs = 200):
    X, y, X_train, X_test, y_train, y_test = data()
    reg = LinearRegression(rate = rate, epochs = epochs)
    reg.fit(X_train, y_train)
    return reg

def plot_slr():
    X, y, X_train, X_test, y_train, y_test = data()
    reg = train_slr()
    fig, ax = plt.subplots()
    line1, = ax.plot(X, reg.w_hist[0] * X + reg.b_hist[0], "aqua", lw = 2)
    line2, = ax.plot(X, reg.w_hist[0] * X + reg.b_hist[0], "white", lw = 1)  
    plt.scatter(X_train, y_train, c = "peru", edgecolors = "black")
    textbox = ax.text(0.02, 0.95, "", transform = ax.transAxes, verticalalignment = "top")
    mse_text = ax.text(0.02, 0.86, "", transform = ax.transAxes, verticalalignment = "top")

    def animate(i):
        line1.set_ydata(reg.w_hist[i] * X + reg.b_hist[i]) 
        line2.set_ydata(reg.w_hist[i] * X + reg.b_hist[i]) 
        textbox.set_text(f'w = {reg.w_hist[i][0]:.2f}\nb = {reg.b_hist[i]:.2f}')
        mse_text.set_text(f'MSE = {reg.mse_hist[i]:.2f}')
        return line1, line2, textbox, mse_text

    ani = animation.FuncAnimation(fig, animate, frames = range(len(reg.w_hist)), interval = 5) 
    plt.show()