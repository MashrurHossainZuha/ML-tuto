# You will use the motivating example of housing price prediction. The training dataset contains
# three examples with four features (size, bedrooms, floors and, age) shown in the table below.
import numpy as np
import matplotlib.pyplot as plt
import math, copy

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# size, num of bedrooms, number of floors, age of home
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])  # training examples
y_train = np.array([460, 232, 178])  # prices

# data is stored in numpy array/matrix
# print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
# print(X_train)
# print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
# print(y_train)

w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])  # w is a parameter vector
b_init = 785.1811367994083  # b is also a parameter


# print(w_init)
# print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


# cost function, efficient multiplication using dot vectors
def compute_cost(X, y, W, b):
    m = X.shape[0]  # number of rows

    cost = 0.0
    for i in range(m):
        f_wb = np.dot(X[i], W) + b
        cost += (f_wb - y[i]) ** 2

    cost /= (2 * m)
    return cost


# cost = compute_cost(X_train, y_train, w_init, b_init)
#
#
# # print(cost)


# gradient descent for multiple variables
def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """

    m, n = X.shape  # number of examples, number of features -> basically rows and cols

    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]  # multiply by X[i, j]th element, similar to matrices indexing
        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)  # Calc the gradient and update the parameters

        # calculations are done in vector notation
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history  # retun final w, b and J_history for graph plotting


# final calculation
# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.0
# gradiant descent setting
iterations = 1000
alpha = 5.0e-7

# Run gradient descent
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient,
                                            alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

m = X_train.shape[0]
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration")
ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()

# costs are not that accurate, we need to improve
