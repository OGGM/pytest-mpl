from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import pytest


def plot(line, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(line, **kwargs)
    return fig


def bar(height, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(height))
    ax.bar(x, height, align="center", **kwargs)
    return fig


def _free_fig(func):
    @wraps(func)
    def free_wrapper(*args, **kwargs):
        fig = func(*args, **kwargs)
        plt.close(fig)
        return fig
    return free_wrapper


def figure_test(test_function, **kwargs):
    @_free_fig
    @pytest.mark.image
    @pytest.mark.hash
    @pytest.mark.mpl_image_compare(savefig_kwargs={'metadata': {'Software': None}}, **kwargs)
    @wraps(test_function)
    def test_wrapper(*args, **kwargs):
        return test_function(*args, **kwargs)
    return test_wrapper
