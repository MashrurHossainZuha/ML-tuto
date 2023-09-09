# Question is to make a linear regression model to the value of 1200 sqft house in 1000s dollar
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])  # size (1000 sqft)
y_train = np.array([300, 500])  # price (1000s dollars)

# m = x_train.shape[0]  # number of training models

plt.title("Housing Prices")  # title of the graph

plt.ylabel("Price (in 1000s of dollars)")  # label of y-axis
plt.ylabel("Size (in 1000 sqft)")  # label of x-axis


# plt.scatter(x_train, y_train)  # puts the points in x, y co-ordinate
# plt.show()  # draws the board


# f(w,b) = wx + b , function for staright line

# computing the trained model

def compute_train_model(x, w, b):
    m = x.shape[0]  # number of modules to train
    f_train = np.zeros(m)  # creates an array of size of m with zeroes
    for i in range(m):
        f_train[i] = w * x[i] + b
    # basically returns the y value as y = mx +_c, so we can put points as (x,y)
    return f_train


# func variables - > we found it by guessing in this stage
w = 200
b = 100
# the purpose was the get the value of w, b which is closest to our main values so it matches the points
# bt in this tuto we just guessed it

temp_f = compute_train_model(x_train, w, b)

plt.plot(x_train, temp_f, c="b", label="Our Prediction")
plt.scatter(x_train, y_train, marker="x", c="r", label="Actual Values")  # puts the points in x, y co-ordinate
plt.show()  # draws the board

w = 200
b = 100

x_i = 1.2  # required value

cost = w * x_i + b
print(f'The value of {x_i * 1000} sqft house is ${cost:.0f} thousand dollars ')
