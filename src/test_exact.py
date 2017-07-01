from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

(x, y) = load_boston(True)

# 厳密解：0.740607742865
lr = LinearRegression()
lr.fit(x, y)
print(lr.score(x, y))
