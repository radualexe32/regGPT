from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from sl_reg import LinearRegression
from logistic_reg import LogisticRegression
from pol_reg import PolynomialRegression

plt.style.use("seaborn-darkgrid")

def data(samples = 100, features = 1, noise = 5, degree = 1):
    X, y = datasets.make_regression(n_samples = samples, n_features = features, noise = noise, random_state = 42)
    y = y**degree
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)
    return X, y, X_train, X_test, y_train, y_test

def train_slr(rate = 0.01, epochs = 200, mini_batch = False):
    X, y, X_train, X_test, y_train, y_test = data()
    reg = LinearRegression(rate = rate, epochs = epochs)
    if not mini_batch:
        reg.fit(X_train, y_train)
    else:
        reg.fit_mini_batch(X_train, y_train)
    print(f"y = {reg.w}x + {reg.b} --> Mini-Batch GD: {mini_batch}")
    return reg

def plot_slr(mini_batch = False):
    X, y, X_train, X_test, y_train, y_test = data()
    reg = train_slr(mini_batch = mini_batch)
    fig, ax = plt.subplots()
    line1, = ax.plot(X, reg.w_hist[0] * X + reg.b_hist[0], "aqua", lw = 3)
    line2, = ax.plot(X, reg.w_hist[0] * X + reg.b_hist[0], "white", lw = 2)  
    plt.scatter(X_train, y_train, c = "peru", edgecolors = "black")
    info_text = ax.text(0.02, 0.95, "", transform = ax.transAxes, verticalalignment = "top")
    stat_text = ax.text(0.19, 0.95, "", transform = ax.transAxes, verticalalignment = "top")
 
    def animate(i):
        line1.set_ydata(reg.w_hist[i] * X + reg.b_hist[i]) 
        line2.set_ydata(reg.w_hist[i] * X + reg.b_hist[i]) 
        info_text.set_text(f"w = {reg.w_hist[i][0]:.2f} \n b = {reg.b_hist[i]:.2f}")
        stat_text.set_text(f"MSE = {reg.mse_hist[i]:.2f} \n R$^2$ = {reg.r2_hist[i]:.2f}")
        return line1, line2, info_text, stat_text

    ani = animation.FuncAnimation(fig, animate, frames = range(len(reg.w_hist)), interval = 10) 
    plt.show()

def train_pol_reg(rate = 0.001, epochs = 1000, degree = 2, mini_batch = False):
    X, y, X_train, X_test, y_train, y_test = data(degree = degree)
    reg = PolynomialRegression(degree = degree, rate = rate, epochs = epochs)
    if not mini_batch:
        reg.fit(X_train, y_train)
    else:
        reg.fit_mini_batch(X_train, y_train)
    print(f"y = {reg.w}x + {reg.b} --> Mini-Batch GD: {mini_batch}")
    return reg

def plot_pol_reg(mini_batch = False):
    X, y, X_train, X_test, y_train, y_test = data(degree = 2)
    reg = train_pol_reg(mini_batch = mini_batch)
    y_hat = reg.predict(X_train)

    plt.plot(X_train, y_hat, "aqua", lw = 3)
    plt.plot(X_train, y_hat, "white", lw = 2)
    plt.scatter(X, y, color = "peru", edgecolors = "black")
    textstr = '\n'.join((
        r'$\mathrm{weights}=%.2f,%.2f,%.2f$' % (reg.w[0], reg.w[1], reg.w[2]), 
        r'$\mathrm{bias}=%.2f$' % (reg.b,), 
        r'$\mathrm{MSE}=%.2f$' % (reg.mse_hist[-1],), 
        r'$\mathrm{R^2}=%.2f$' % (reg.r2_hist[-1],)))
    props = dict(boxstyle = "round", facecolor = "wheat", alpha=0.5)
    plt.gca().text(0.45, 0.95, textstr, transform = plt.gca().transAxes, fontsize = 10,
        verticalalignment = "top", bbox = props)
    plt.show()
