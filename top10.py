import numpy as np
def take10encodings(encodings, names):
	mean = np.sum(encodings, axis=0)/encodings.shape[1]
	#m = mean*np.ones(encodings.shape[1])
	var = np.sum(abs((encodings-mean)),axis=1)
	top10en = encodings[np.argsort(var)[-10:]]
	top10names = names[np.argsort(var)[-10:]]
	return top10, top10names

