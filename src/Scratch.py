from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas
import GPy
import matplotlib.pyplot as plt
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pandas.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

T = df.loc[:,['target']].values
Y = df.loc[:, features].values
Y = StandardScaler().fit_transform(Y)
N = np.size(Y, axis=0)
D = np.size(Y, axis=1)

beta = 10
PCA_N = 2

X = np.zeros((N, PCA_N))
W = np.zeros((D, PCA_N))
# print(np.dot(X, W.T))



pca = PCA(n_components=2)
principalComponents = pca.fit_transform(Y)
print(principalComponents.shape)
principalDf = pandas.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pandas.concat([principalDf, df[['target']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'm', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()