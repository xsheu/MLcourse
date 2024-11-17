# 請不要動
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
subprocess.check_call([sys.executable, "-m","pip", "install", "numpy"])
subprocess.check_call([sys.executable, "-m","pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m","pip", "install", "scikit-learn"])
# 請不要動
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score

# 讀取資料檔案
data =  pd.read_csv('financial_ratio.csv')
print(data.head())
total_sample=data.shape[0]
# 資料筆數
print(total_sample)
# 指定X與Y
X = data.iloc[:,0:4].values
Y = data['class'].values
# 分割測試與驗證資料
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = .25,random_state = 0)
# 正規化
stdscaler = StandardScaler()
X_train = stdscaler.fit_transform(X_train)
X_test = stdscaler.transform(X_test)
# 建立支援向量機向量機模式
rf =  RandomForestClassifier(random_state=0)
rf.fit(X_train,Y_train)
y_pred=rf.predict(X_test)
print("Accuracy")
print(accuracy_score(Y_test,y_pred))
print("Precision")
print(precision_score(Y_test,y_pred))
print("Recall")
print(recall_score(Y_test,y_pred))
print("Specificity")
print(recall_score(Y_test,y_pred,pos_label=0))
print("F1 score")
print(f1_score(Y_test,y_pred))
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
auc_score=roc_auc_score(Y_test, y_pred)
print("AUC")
print(auc_score)
print(f'Classification Report: \n{classification_report(Y_test, y_pred)}')
cm = confusion_matrix(Y_test, y_pred)
print("Confusion matrix")
print(cm)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(fpr, tpr)
plt.show()
