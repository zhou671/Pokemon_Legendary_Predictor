# data.py

######################################
# data - Extract Neccessary Data
#
#####################################

import numpy as np
import random

def run():
	N = 800
	datap =  np.load('x_clean.npy')
	count = 1

	while (count < 9) :
		tempData = np.array(random.sample(datap, 100))

		if ((tempData==-1).sum() != 8 & (tempData==-1).sum() != 9):
			tempData = np.array(random.sample(datap, 100))
			print 'not good'
		
		else:
			print 'good'
			xfile = "./dataset"+ str(count) + "_X"
			yfile = "./dataset"+ str(count) + "_Y"
			np.save(xfile, tempData[:, 1:7])
			np.save(yfile, tempData[:, 7])

			for i in range(100):
				np.delete(datap, tempData[i][0])
		
			count = count + 1

if __name__ == '__main__':
	run() 