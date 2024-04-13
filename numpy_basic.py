import numpy as np

# Create a 1D array
a = np.array([1, 2, 3, 4, 5])
print(a)
# create basic mathematical functions

def add_10(x):
    return x + 10

def add_array(a):
    return np.array([add_10(x) for x in a])

def add_number(a, n):
    return np.array([x + n for x in a])

def multiply_array(a, b):
    return np.array([a[i] * b[i] for i in range(len(a))])

def multiply_number(a, n):
    return np.array([x * n for x in a])

def divide_array(a, b):
    return np.array([a[i] / b[i] for i in range(len(a))])
