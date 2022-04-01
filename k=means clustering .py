from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
x=df.iloc[:,0:2]
y=iris.target
km=KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)
y_km=km.fit_predict(x)
#displaying the cluster centers
print(km.cluster_centers_)
#plotting the clusters
import matplotlib.pyplot as plt
plt.scatter(x.iloc[y_km==0,0],x.iloc[y_km==0,1],s=100,c='red',label='Iris-setosa')
plt.scatter(x.iloc[y_km==1,0],x.iloc[y_km==1,1],s=100,c='blue',label='Iris-versicolor')
plt.scatter(x.iloc[y_km==2,0],x.iloc[y_km==2,1],s=100,c='purple',label='Iris-virginica')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=200,c='yellow',label='Centroids')
plt.legend()
plt.show()
#print the number of steps to stabilize the model
print(km.n_iter_)
