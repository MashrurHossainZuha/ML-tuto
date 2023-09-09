import numpy as np
import matplotlib.pyplot as plt
import math, copy

# Load our data set
x_train = np.array([1.0, 2.0])  # features
y_train = np.array([300.0, 500.0])  # target value


# implement cost function

def cost_function(x, y, w, b):
    m = x.shape[0]
    sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        sum += (f_wb - y[i]) ** 2
    return sum / (2 * m)


# compute gradient -> partial derivative of j(w,b) w.r.t w and b respectively

def compute_gradient(x, y, w, b):
    dj_dw = 0
    dj_db = 0

    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])

    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    """

    J_history, p_history = [], []
    w, b = w_in, b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 10000:  # prevent exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history  # returns best approximate w, b and J, p history for graph plotting


# initialize paramenters
w_init = 0
b_init = 0

# gradient descent const values
iters = 10000
tmp_alpha = 1.0e-2

# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iters)

# final value of w, b
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")



# predictoins
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")