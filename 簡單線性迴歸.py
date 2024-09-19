# 請不要動下列四行
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m","pip", "install", "numpy"])
# 請不要動上述四行
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [10, 80], [8, 0], [8, 200], [5, 200], [7, 300], [8, 230], [7, 40], [9, 0], [6, 330], [9, 180]
])
y = np.array([469, 366, 371, 208, 246, 297, 363, 436, 198, 364])
lm = LinearRegression()
lm.fit(X, y)
print(lm.coef_)
print(lm.intercept_ )
