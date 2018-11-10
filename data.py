# data.py

######################################
# data - Extract Neccessary Data
#
#####################################

import numpy as np
#import random

def run():
	data =  np.load('inputData\\x_clean.npy')
	
	## delete id colomn
	data = data[:, 1:8]

	## shuffle 800 
	np.random.shuffle(data)
	np.random.shuffle(data)

	## seperate data into negative and positive labels
	data_pos = data[np.where(data[:,6] == 1)]
	data_neg = data[np.where(data[:,6] == -1)]

	## shuffle 735 positive samples
	np.random.shuffle(data_pos)
	np.random.shuffle(data_pos)

	## shuffle 65 positive samples
	np.random.shuffle(data_neg)
	np.random.shuffle(data_neg)

	## X0-X6, Y0-Y6
	for i in range(7):
		namex = "data\\X" + str(i) + ".npy"
		namey = "data\\Y" + str(i) + ".npy"
		
		x = np.zeros((100, 6))
		y = np.zeros((100))

		x[0:92] = data_pos[i*92:i*92+92, 0:6]
		y[0:92] = data_pos[i*92:i*92+92, 6]
		x[92:100] = data_neg[i*8:i*8+8, 0:6]
		y[92:100] = data_neg[i*8:i*8+8, 6]

		np.save(namex, x)
		np.save(namey, y)

	## X7, Y7
	namex = "data\\X7.npy"
	namey = "data\\Y7.npy"
	x = np.zeros((100, 6))
	y = np.zeros((100))
	x[0:91] = data_pos[644:735, 0:6]
	y[0:91] = data_pos[644:735, 6]
	x[91:100] = data_neg[56:65, 0:6]
	y[91:100] = data_neg[56:65, 6]
	np.save(namex, x)
	np.save(namey, y)

if __name__ == '__main__':
	run() 