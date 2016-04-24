import sys
import random
from collections import Counter
import math

def generate_samples(filename):
	with open(filename,"wb") as f:
		for i in range(1000):
			f.write(str(random.gauss(0,2)))
			f.write("\t0\n")
		for i in range(1000):
			f.write(str(random.gauss(5,1)))
			f.write("\t1\n")

def get_density(point, mean, stdevs):
	return (1/(stdevs* math.sqrt(2 * math.pi))) * (math.pow(math.e, -(point-mean)*(point-mean) / (2*stdevs*stdevs)))

def get_posterior(point, priors, means, stdevs):
	posteriors = Counter()
	for k in priors.keys():
		posteriors[k] = get_density(point, means[k], stdevs[k]) * priors[k]

	sum_posteriors = sum(posteriors.values())
	for k in posteriors.keys():
		posteriors[k] /= sum_posteriors
	
	return posteriors

def relearn(plines):
	priors = Counter()
	means  = Counter()
	stdevs = Counter()

	for (posterior, line) in plines:
		for k in posterior.keys():
			priors[k] += posterior[k]
			means[k]  += posterior[k] * float(line[0])
			stdevs[k] += posterior[k] * float(line[0]) * float(line[0])

		sum_priors = sum(priors.values())
		for k in priors.keys():
			means[k] /= priors[k]
			stdevs[k]/= priors[k]
			stdevs[k] = math.sqrt(stdevs[k] - means[k] * means[k])
			priors[k]/= sum_priors

		return (priors, means, stdevs)

def main():
	op = sys.argv[1]
	trainfile = "../Output/gmm/gmm.test"
	outputfile= "../Output/gmm/gmm.output"
	f = open(outputfile,"wb")
	if op == "generate":
		generate_samples(trainfile)
	else:
		priors = Counter({"0": 0.5, "1":0.5})
		means  = Counter({"0": random.random(), "1":random.random()})
		stdevs = Counter({"0": random.random(), "1":random.random()})

		for i in range(1000):
			lines = [l.strip().split() for l in open(trainfile).readlines()]
			plines= []
			for line in lines:
				plines.append((get_posterior(float(line[0]), priors, means, stdevs), line))

			f.write("Iterations: "+str(i)+"\n")
			f.write("Priors: "+str(priors)+"\n")
			f.write("Means: "+str(means)+"\n")
			f.write("Stdevs: "+str(stdevs)+"\n")
			(priors, means, stdevs) = relearn(plines)
	f.close()

if __name__ == '__main__':
	main()