# square mean error counting

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')


x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2]) # (size in 1000 square feet)
y_train = np.array([250, 300, 480,  430,   630, 730,])  # (price in 1000 of dollars)

# cost function
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        total_cost += (f_wb - y[i]) ** 2  # square mean error

    return total_cost / (2 * m)



