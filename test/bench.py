import sys
import os
import timeit
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))

from slr import * 
from lor import *
from basis_functions import *
from train import * 

if __name__ == "__main__":
    start = timeit.default_timer()
    train_slr()
    end = timeit.default_timer()
    print(f"Time taken: {(end - start) * 1000} ms")