import autograd.numpy as np
from autograd import grad
from sklearn.datasets import load_boston

(x, y) = load_boston(True)
ss_total = ((y - y.mean()) ** 2).sum()


def evaluate(p):
    return ((np.inner(x, p[:-1]) + p[-1] - y) ** 2).sum() / ss_total


def evaluate_one(p, xi, yi):
    return ((np.inner(xi, p[:-1]) + p[-1] - yi) ** 2) / ss_total


grad_evaluate = grad(evaluate_one)

# AdaGrad (stochastic)
learn_rate = 0.3
p = np.random.randn(14)
r = np.full([14], 1e-8)
indices = np.arange(len(y))
for j in range(10000):
    np.random.shuffle(indices)
    for i in indices:
        d = grad_evaluate(p, x[i], y[i])
        r += d * d
        p -= learn_rate / np.sqrt(r) * d
    print("%d: r^2 = %.12f" % (j, 1 - evaluate(p)))
