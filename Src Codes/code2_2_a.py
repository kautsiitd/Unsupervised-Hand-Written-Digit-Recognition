import os
import csv
import scipy.io
import scipy.misc
import numpy as np
from random import shuffle
from sklearn.cluster import KMeans as km

# defining variables
n_cluster = 5
c_label	  = [0]*n_cluster
result 	  = [[] for i in range(n_cluster)]

# loading data
mat 	= scipy.io.loadmat('../dataset/2012ME20780.mat')
data 	= mat["data_image" ]
labels 	= mat["data_labels"]
frqs	= [['Actual/Predict',0,1,2,3,4,5,6,7,8,9,'Recall']]+[[i]+[0 for j in range(10)] for i in range(10)]+[['Precision']]

# saving original pics
for i in range(2000):
	scipy.misc.imsave("../Output/2.2a/Original_"+str(n_cluster)+"/"+str(i)+".bmp",data[i].reshape(28,28))

# modeling and predicting labels
model	= km(n_clusters=n_cluster, max_iter=500, n_init=100, init='k-means++', tol=.00001, n_jobs=4)
model.fit(np.array(data))
print 1
p_label = model.fit_predict(data)
print 2

# finding mapping
for i in range(2000):
	result[p_label[i]].append(labels[i][0])
mapping = [max(set(i), key=i.count) for i in result]

# saving results
for i in range(2000):
	dir_name = "../Output/2.2a/Cluster_"+str(n_cluster)+"/"+str(mapping[p_label[i]])
	try:os.mkdir(dir_name)
	except:pass
	scipy.misc.imsave(dir_name+"/"+str(c_label[p_label[i]])+".bmp",data[i].reshape(28,28))
	c_label[p_label[i]]	+= 1
	frqs[mapping[p_label[i]]+1][labels[i]+1] += 1

for i in range(1,11):
	frqs[ i].append(round((frqs[i][i]*100.0)/sum(frqs[i]),2))
	frqs[-1].append(round((frqs[i][i]*100.0)/sum(zip(*frqs[1:-1])[i]),2))

with open("../Output/2.2a/Cluster_"+str(n_cluster)+"/Result.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(frqs)