'''
圖解： https://cdn-images-1.medium.com/max/1000/1*KrcZK0xYgTa4qFrVr0fO2w.gif
'''
from sklearn.cluster import KMeans
import numpy as np

# 訓練資料
data = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])

# 訓練模型
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
print('labels=', kmeans.labels_)
print('centers=', kmeans.cluster_centers_)
print('predict [1.5, 2.5] = ', kmeans.predict([[1.5, 2.5]]))


# get array of label=0
x0=[]
y0=[]
for x,y in [data[index] for index, x in enumerate(kmeans.labels_) if x == 0]:
    x0.append(x)
    y0.append(y)

# get array of label=1
x1=[]
y1=[]
for x,y in [data[index] for index, x in enumerate(kmeans.labels_) if x == 1]:
    x1.append(x)
    y1.append(y)

# get array of label=1
x2=[]
y2=[]
for index, data in enumerate(kmeans.cluster_centers_):
    x2.append(data[0])
    y2.append(data[1])

# 畫圖              
import matplotlib.pyplot as plt
plt.scatter(x0, y0, color='g')
plt.scatter(x1, y1, color='r')
plt.scatter(x2, y2, color='b')
plt.show()
