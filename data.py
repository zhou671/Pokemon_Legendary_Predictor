# data.py

######################################
# data - Extract Neccessary Data
#
#####################################

import numpy as np

def run():
	x = np.load("x.npy")
	N = len(x)

	# construct features
	X = x[:, 2:8]
	np.save("./features", X)

	# construct labels
	Y = np.ones([N, 1])
	for i in range(N):
		if (x[i][9] == 1):
			Y[i] = -1

	np.save("./labels", Y)

	data = np.hstack((X, Y))
	np.random.shuffle(data)
	np.random.shuffle(data)

	#dataset 1
	np.save("./dataset1_X", data[0:100, 0:6])
	np.save("./dataset1_Y", data[0:100, 6])

	#dataset 2
	np.save("./dataset2_X", data[100:200, 0:6])
	np.save("./dataset2_Y", data[100:200, 6])

	#dataset 3
	np.save("./dataset3_X", data[200:300, 0:6])
	np.save("./dataset3_Y", data[200:300, 6])

	#dataset 4
	np.save("./dataset4_X", data[300:400, 0:6])
	np.save("./dataset4_Y", data[300:400, 6])

	#dataset 5
	np.save("./dataset5_X", data[400:500, 0:6])
	np.save("./dataset5_Y", data[400:500, 6])

	#dataset 6
	np.save("./dataset6_X", data[500:600, 0:6])
	np.save("./dataset6_Y", data[500:600, 6])

	#dataset 7
	np.save("./dataset7_X", data[600:700, 0:6])
	np.save("./dataset7_Y", data[600:700, 6])

	#dataset 8
	np.save("./dataset8_X", data[700:800, 0:6])
	np.save("./dataset8_Y", data[700:800, 6])



if __name__ == '__main__':
	run() 