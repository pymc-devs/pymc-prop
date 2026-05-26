import numpy as np


def pytest_configure():
    np.set_printoptions(precision=6, suppress=True)
