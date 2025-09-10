# 請不要動下列六行
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m","pip", "install", "numpy"])
subprocess.check_call([sys.executable, "-m","pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m","pip", "install", "matplotlib"])
# 請不要動上述六行
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X,y=make_regression(n_samples=100, n_features=1, noise=50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

plt.scatter(X,y)

regr=linear_model.LinearRegression()
regr.fit(X_train, y_train)

plt.scatter(X_train, y_train, color='black')
plt.plot(X_train, regr.predict(X_train),color='blue',linewidth=1)
plt.show()

w_0=regr.intercept_
w_1=regr.coef_

print(w_0)
print(w_1)

regr.score(X_train,y_train)

print(regr.score(X_train,y_train))
