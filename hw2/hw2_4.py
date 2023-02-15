# %%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import os
from os import path

input_loc = path.join(path.dirname(__file__)) # Folder

a = 0
b = 10
x = np.linspace(a, b, num=100)
y = np.sin(x)

# Train a model with input noise
n = 20 # number of sampled elements
x_train_no_noise = np.sort(np.random.choice(x, n, replace=False))
x_train = np.sort(np.add(x_train_no_noise, np.random.normal(0, 0.5, size=n)))
y_train = np.sin(x_train)
poly = lagrange(x_train, y_train)

# Test the trained model without input noise
x_test = np.sort(np.random.choice(x, n, replace=False)) # random select from x
y_test = np.sin(x_test)
y_pred = Polynomial(poly.coef[::-1])(x_test)

# Calculating train and test error
train_error = y_pred - y_train
test_error = y_pred - y_test

# Print out the errors in form of standard deviation
print('Train error: ',np.std(train_error))
print('Test error: ', np.std(test_error))

# Plot the results
plt.figure(figsize=(5, 3.5), dpi=100)
plt.scatter(x, y, label='original', c='grey', s=5)
plt.plot(x_train, y_train, label='train', c='r')
plt.plot(x_test, y_test, label='test', c='g')
plt.ylim([-1.5, 1.5])
plt.legend()
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig(input_loc + '/figures/hw2_4.pdf', format='pdf', bbox_inches='tight') 
plt.show()

# %%
