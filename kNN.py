# 請不要動
import subprocess
import sys
subprocess.check_call([sys.executable, "-m","pip", "install", "numpy"])
subprocess.check_call([sys.executable, "-m","pip", "install", "matplotlib"])
subprocess.check_call([sys.executable, "-m","pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m","pip", "install", "scikit-learn"])
# 請不要動

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("prostate.csv")
print(df.head())


from sklearn.preprocessing import StandardScaler

y=df['Target'].values
x = X = df.iloc[:,0:7].values

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
 
X_train, X_test,y_train, y_test = train_test_split(x,y,test_size=0.30)

stdscaler = StandardScaler()
X_train = stdscaler.fit_transform(X_train)
X_test = stdscaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


error_rate = []
 
# Will take some time
for i in range(1, 40):
 
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
 
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue',
         linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
 
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

knn = KNeighborsClassifier(n_neighbors = 1)
 
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
 
print('WITH K = 1')
print('Confusion Matrix')
print(confusion_matrix(y_test, pred))
print('Classification Report')
print(classification_report(y_test, pred))

knn = KNeighborsClassifier(n_neighbors = 10)
 
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
 
print('WITH K = 10')
print('Confusion Matrix')
print(confusion_matrix(y_test, pred))
print('Classification Report')
print(classification_report(y_test, pred))
