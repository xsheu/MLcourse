# 請不要動
import subprocess
import sys
subprocess.check_call([sys.executable, "-m","pip", "install", "numpy"])
subprocess.check_call([sys.executable, "-m","pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m","pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m","pip", "install", "matplotlib"])
subprocess.check_call([sys.executable, "-m","pip", "install", "seaborn"])
# 請不要動

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_sms = pd.read_csv('spam.csv',encoding='latin-1')
print(df_sms.head())
df_sms = df_sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df_sms = df_sms.rename(columns={"v1":"label", "v2":"sms"})
print(df_sms.head())
df_sms['Labeling']= df_sms['label'].map({'ham': 1, 'spam':0})
print(df_sms.head())
df_sms['length'] = df_sms['sms'].apply(len)
print(df_sms.head())


df_sms['length'].plot(bins=50, kind='hist')

df_sms.hist(column='length', by='label', bins=50,figsize=(10,4))
plt.show()

X = df_sms['sms']
Y = df_sms['Labeling']

from sklearn.model_selection import train_test_split as tt
X_train, X_test, Y_train, Y_test = tt(X, Y,test_size=0.25, random_state=100)
X_trainlist=X_train.tolist()
X_testlist=X_test.tolist()
Y_trainlist=Y_train.tolist()
Y_testlist=Y_test.tolist()

print(X_train.shape)

from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(stop_words ='english')
vector.fit(X_train)

X_train_transformed =vector.transform(X_train)
X_test_transformed =vector.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_transformed,Y_train)
y_pred = model.predict(X_test_transformed)
y_pred_prob = model.predict_proba(X_test_transformed)

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
print("Precision=",precision_score(Y_test,y_pred))
print()
print("Recall=",recall_score(Y_test,y_pred))
print()
print("F1 score=",f1_score(Y_test,y_pred))
print()
print("Accuracy=",accuracy_score(Y_test,y_pred))

from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred_prob[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC=",roc_auc)
