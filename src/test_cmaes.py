import cma
import numpy as np
from sklearn.datasets import load_boston

(x, y) = load_boston(True)
ss_total = ((y - y.mean()) ** 2).sum()


def evaluate(p):
    return ((np.inner(x, p[:-1]) + p[-1] - y) ** 2).sum() / ss_total


res = cma.fmin(evaluate, np.random.randn(14), 1, {'verb_log': False})
print("r^2 = %.12f" % (1 - res[1]))
