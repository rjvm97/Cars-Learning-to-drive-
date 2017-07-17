import pickle
import numpy as np 

np.random.seed(1)
i = list()
for j in range(10):
	syn0 = 2*np.random.random((8,4)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
	syn1 = 2*np.random.random((4,2)) - 1 
	i.append([syn0, syn1])
arq = open("individuos.pickle", "wb")
pickle.dump(i, arq)