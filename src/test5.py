import numpy as np
from sklearn.datasets import load_boston

(x, y) = load_boston(True)
ss_total = ((y - y.mean()) ** 2).sum()


def evaluate(p):
    return 1 - ((np.inner(x, p[:-1]) + p[-1] - y) ** 2).sum() / ss_total


iter_count = 10000
pop_size = 100
child_size = 30
params_len = 14
tau = 1.0 / np.sqrt(2.0 * params_len)

with open("../result/5_%d.tsv" % iter_count, "wt") as fp:
    for iter in range(100):
        print(iter)
        individuals = []
        for i in range(pop_size):
            params = np.random.randn(params_len)
            strategy = 0.1
            individuals.append((params, strategy, evaluate(params)))
        individuals = sorted(individuals, key=lambda ind: -ind[2])
        best = individuals[0]

        for _ in range(iter_count):
            for _ in range(child_size):
                # 交叉
                if np.random.rand() < 0.8:
                    ind0 = individuals[np.random.randint(pop_size)]
                    ind1 = individuals[np.random.randint(pop_size)]
                    r = np.random.randint(0, 2, [params_len])
                    parent = (ind0[0] * r + ind1[0] * (1 - r), (ind0[1] + ind1[1]) / 2.0)
                else:
                    parent = individuals[np.random.randint(pop_size)]

                # 突然変異
                strategy2 = parent[1] * np.exp(tau * np.random.randn())
                params2 = parent[0] + strategy2 * np.random.randn()
                individuals.append((params2, strategy2, evaluate(params2)))

            individuals = sorted(individuals, key=lambda ind: -ind[2])[:pop_size]
            best = individuals[0]
        print(best[2], file=fp, flush=True)
