import pandas as pd
import numpy as np
#loading breast cancer dataset
url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
features=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nuclei','Mitoses','Class']
df=pd.read_csv(url,names=features)
df.drop(['Sample code number'],axis=1,inplace=True)
#print(df['Bare Nuclei'].unique())
df=df.replace('?',np.nan)
print(df.isnull().sum())
df['Bare Nuclei']=df['Bare Nuclei'].fillna(df['Bare Nuclei'].median())
print(df.dtypes)
df['Bare Nuclei']=df['Bare Nuclei'].astype(int)

x=df.drop(['Class'],axis=1)
y=df['Class']

from sklearn import svm
clf=svm.SVC(kernel='linear')
from sklearn.model_selection import KFold
kf=KFold(n_splits=10)
from sklearn.metrics import f1_score
kf1=[]
for train_index,test_index in kf.split(x):
    x_train,x_test=x.iloc[train_index],x.iloc[test_index]
    y_train,y_test=y.iloc[train_index],y.iloc[test_index]
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    kf1.append(f1_score(y_test,y_pred,pos_label=2))
print(kf1)
print('Mean f1 score of 10 fold validation',np.mean(kf1))

from sklearn.model_selection import train_test_split
f1=[]
r=0.2,0.3,0.4
for i in r:
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=i)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    f1.append(f1_score(y_test,y_pred,pos_label=2))

print(f1)

#comparison table of f1 score
import matplotlib.pyplot as plt
plt.plot(r,f1)
plt.xlabel('Test size')
plt.ylabel('F1 score')
plt.show()
