import numpy as np
import matplotlib.pyplot as plt

# from lab_utils_common import plot_data, sigmoid, dlc

X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  # (m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])  # (m,)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(w, X[i]) + b
        f_wb = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1-f_wb) # formula
    cost /= m
    return cost


w_tmp = np.array([1, 1])
b_tmp = -3

print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))