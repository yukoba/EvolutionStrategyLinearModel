import autograd.numpy as np
from autograd import grad
from sklearn.datasets import load_boston

(x, y) = load_boston(True)
ss_total = ((y - y.mean()) ** 2).sum()


def evaluate(p):
    return ((np.inner(x, p[:-1]) + p[-1] - y) ** 2).sum() / ss_total


grad_evaluate = grad(evaluate)

# AdaGrad (non-stochastic)
learn_rate = 1.0
p = np.random.randn(14)
r = np.full([14], 1e-8)
for j in range(10000):
    d = grad_evaluate(p)
    r += d * d
    p -= learn_rate / np.sqrt(r) * d
    print("%d: r^2 = %.12f" % (j, 1 - evaluate(p)))
