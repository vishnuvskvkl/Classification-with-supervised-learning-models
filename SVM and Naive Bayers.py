import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
features=(["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"])
df=pd.read_csv(url,names=features)
df.drop(["Sample code number","Bare Nuclei"],axis=1,inplace=True)
print(df.describe())
x=df.drop("Class",axis=1)
y=df["Class"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn import svm
clf=svm.SVC(kernel="linear",C=1)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

from sklearn.metrics import confusion_matrix
y_pred=clf.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
print('accuracy:',accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred,pos_label=2))
print("Recall:",recall_score(y_test,y_pred,pos_label=2))
print("F1 Score:",(f1_score(y_test,y_pred,pos_label=2)*100))
from yellowbrick.classifier import ClassificationReport
visualizer=ClassificationReport(clf)
visualizer.fit(x_train,y_train)
visualizer.score(x_test,y_test)
visualizer.show()


#naive bayes classifier
print("Naive Bayes")
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
y_pred=clf.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
print('accuracy:',accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred,average="micro"))
print("Recall:",recall_score(y_test,y_pred,average="micro"))
print("F1 Score:",f1_score(y_test,y_pred,average="micro"))
