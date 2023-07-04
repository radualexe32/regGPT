from train import *


if __name__ == "__main__":
    plot_slr(mini_batch=True)
    plot_slr(mini_batch=False)
    plot_pol_reg(mini_batch=True)
    plot_pol_reg(mini_batch=False)
