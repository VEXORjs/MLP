import math
import time
<<<<<<< HEAD
=======

>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
from pathlib import Path


def ensure_dir(path):
<<<<<<< HEAD

    if not path:
        path = "."

    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def sigmoid(x):

=======
    Path(path).mkdir(parents=True, exist_ok=True)

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def sigmoid(x):
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
    if x < -700:
        return 0.0

    if x > 700:
        return 1.0

    return 1.0 / (1.0 + math.exp(-x))

<<<<<<< HEAD

def sigmoid_derivative_from_output(y):
    return y * (1.0 - y)


def mse(targets, outputs):

=======
def sigmoid_derivative_from_output(y):
    return y * (1.0 - y)

def mse(targets, outputs):
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
    s = 0.0

    for t, o in zip(targets, outputs):

        d = t - o
        s += d * d

    return 0.5 * s

<<<<<<< HEAD

def one_hot(index, size):

    arr = [0.0] * size
    arr[index] = 1.0

    return arr
=======
def one_hot(index, size):
    arr = [0.0] * size
    arr[index] = 1.0

    return arr

def flatten_index(row, col, cols):
    return row * cols + col
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
