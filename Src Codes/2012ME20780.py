import sys
import os
from sklearn import KMeans
import random
import math
from collections import Counter

def find_prob(point, mean, stdevs):
	return (1/(stdevs* math.sqrt(2 * 3.14))) * (math.pow(math.e, -(point-mean)*(point-mean) / (2*stdevs*stdevs)))

def get_posterior(point, priors, means, stdevs):
	posteriors = Counter()
	posteriors = [find_prob(point, means[k], stdevs[k]) * priors[k] for k in priors.keys()]
	sum_posteriors = sum(posteriors.values())
	posteriors = [posteriors[k]/sum_posteriors for k in posteriors.keys()]	
	return posteriors

def relearn(plines):
	priors,means,stdevs = Counter(),Counter(),Counter()
	for (posterior, line) in plines:
		priors = [priors[k]+posterior[k] for k in posterior.keys()]
		means  = [means[k]+float(line[0])*posterior[k] for k in posterior.keys()]
		stdevs = [stdevs[k]+pow(float(line[0]),2)*posterior[k] for k in posterior.keys()]
		sum_priors = sum(priors.values())
		means  = [means[k]/priors[k] for k in priors.keys()]
		stdevs = [stdevs[k]/priors[k] for k in priors.keys()]
		stdevs = [math.sqrt(stdevs[k] - means[k] * means[k]) for k in priors.keys()]
		priors = [priors[k]/sum_priors for k in priors.keys()]
		return (priors, means, stdevs)

op = sys.argv[1]
trainfile = "../Output/gmm/gmm.test"
outputfile= "../Output/gmm/gmm.output"
f = open(outputfile,"wb")
if op == "generate":
	with open(filename,"wb") as f:
		(f.write(str(random.gauss(0,2))" 0\n") for i in range(1000))
		(f.write(str(random.gauss(5,1))+" 1\n") for i in range(1000))
else:
	priors = Counter({"0": 0.5, "1":0.5})
	stdevs = Counter({"0": random.random(), "1":random.random()})
	means  = Counter({"0": random.random(), "1":random.random()})

	for i in range(1000):
		lines = [l.strip().split() for l in open(trainfile).readlines()]
		plines= [(get_posterior(float(line[0]), priors, means, stdevs), line) for line in lines]
		f.write("Means: "+str(means)+"\n"+"Stdevs: "+str(stdevs)+"\n")
		(priors, means, stdevs) = relearn(plines)
f.close()
