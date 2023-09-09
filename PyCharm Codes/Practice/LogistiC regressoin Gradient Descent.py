import copy, math
import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(w, X[i]) + b
        f_wb = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)  # formula
    cost /= m
    return cost


def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape  # (m,n) -> (number of parameters, number of features)

    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        z_i = np.dot(w, X[i]) + b
        f_wb = sigmoid(z_i)

        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha, iter):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(iter):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 10000:
            J_history.append(compute_cost_logistic(X, y, w, b))

        if i % math.ceil(iter / 10) == 0 == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history


w_tmp = np.zeros_like(X_train[0])
b_tmp = 0.0
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

