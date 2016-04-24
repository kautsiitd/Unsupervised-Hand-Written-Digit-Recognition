import sys
import os
import csv
import scipy.io
import scipy.misc
import numpy as np
from random import shuffle
from sklearn.cluster import KMeans as km
from sklearn.decomposition import PCA
from sklearn.mixture import GMM
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import affinity_propagation as AP
from sklearn.cluster import DBSCAN
from multiprocessing import Pool
from sklearn.cluster import SpectralClustering as SC
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering as AC

print "import done"

# defining variables
n_cluster = 16

# loading data
data = [map(float,line.split()) for line in open('../dataset/ass3_data.txt')]

# using PCA
# pca 	= PCA(n_components = 25)
# data 	= pca.fit_transform(data)
# sys.exit()

# modeling and predicting labels
	# using Agglomerative Clustering
# model   = AC(n_clusters = 17)
# p_label = model.fit_predict(data)

	# using kmeans
# pca 	= PCA(n_components = 20)
# model	= km(n_clusters=n_cluster, max_iter=1000, n_init=100, init='k-means++', tol=.00001, n_jobs=4)
# pca.fit(data)
# model.fit(np.array(pca.components_))
# print 1
# p_label = model.fit_predict(data)
# print 2
	# using GMM
def f(i):
	global data
	pca 	= PCA(n_components = i)
	temp 	= pca.fit_transform(data)
	model1		  = GMM(n_components = 18, n_iter = 1000, covariance_type = 'full', tol = .000001, min_covar = .000001, n_init = 5, verbose = 1)
	p_label		  = model1.fit_predict(temp)
	print len(p_label)
	# writing in file
	with open('../Output/3_'+str(i)+'.txt','wb') as f:
		for i in p_label:
			f.write(str(i)+',')
	f.close()

	# using hierarcial clustering
# def f(start):
# 	for i in range(5):
# 		thresh = start+float(i)/10
# 		p_label = hcluster.fclusterdata(data, thresh, criterion="distance")
# 		print p_label
# 		print thresh,len(set(p_label))

	# using affinity propogation
# def f():
# 	model  = AP(verbose = 1)
# 	model  = model.fit(data)
# 	p_label= model.predict(data)
# 	# writing in file
# 	with open('Output/3/affinity.txt','wb') as f:
# 		for i in p_label:
# 			f.write(str(i)+',')
# 	f.close()
# f()

	# using DBSCAN
# model   = DBSCAN(min_samples = 18)
# p_label = model.fit_predict(data)

	# using Spectral Clustering
# model   = SC(n_clusters = 17)
# p_label = model.fit_predict(data)

# computing in parrallel
pool = Pool(processes = 2)
pool.map(f, [23,24,25,26,30])

# p_label = hcluster.fclusterdata(data, 7.1, criterion="distance")

# writing in file
# with open('Output/3/3_16.txt','wb') as f:
# 	for i in p_label:
# 		f.write(str(i)+',')
# f.close()