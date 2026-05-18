import math
import time
from pathlib import Path


def ensure_dir(path):

    if not path:
        path = "."

    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def sigmoid(x):

    if x < -700:
        return 0.0

    if x > 700:
        return 1.0

    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative_from_output(y):
    return y * (1.0 - y)


def mse(targets, outputs):

    s = 0.0

    for t, o in zip(targets, outputs):

        d = t - o
        s += d * d

    return 0.5 * s


def one_hot(index, size):

    arr = [0.0] * size
    arr[index] = 1.0

    return arr