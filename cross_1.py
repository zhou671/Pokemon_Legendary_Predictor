# preprocess.py
######################################
# preprocess - remove all the negative one in a training set
#
#####################################
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from itertools import combinations

# cross validation for 700:100
def run1():
	#classifiers = [
	#	(
	#		'gb3',
	#		GradientBoostingClassifier(learning_rate=0.3, n_estimators=110, max_depth=3)
	#	),
	#		'gb4',
	#		GradientBoostingClassifier(learning_rate=0.3, n_estimators=110, max_depth=4)
	#	),
	#	(
	#		'gb4'
	#		GradientBoostingClassifier(learning_rate=0.3, n_estimators=110, max_depth=5)
	#	)
	#]
	clf = GradientBoostingClassifier(learning_rate=0.3, n_estimators=110, max_depth=3)
	List_x = [None] * 8
	List_y = [None] * 8

	x_name = str("dataSets/dataset1_X.npy")
	y_name = str("dataSets/dataset1_Y.npy")

	for i in range(8):
		List_x[i] = np.load(x_name)
		List_y[i] = np.load(y_name)

		x_name = "dataSets/dataset" + str(i + 2) + "_X.npy"
		y_name = "dataSets/dataset" + str(i + 2) + "_Y.npy"

	count = 0

	for i in range(8):
		test_x = List_x[i]
		test_y = List_y[i]

		x = np.zeros((0, 6))
		y = np.zeros((0))

		for j in range(8):
			if not i == j:
				x = np.concatenate((x, List_x[j]), axis = 0)
				y = np.concatenate((y, List_y[j]), axis = None)

		clf.fit(x,y)
		result = clf.predict(test_x)
		
		for j in range(100):
			if test_y[j] == result[j]:
				count = count + 1

	print(count/800.0)
# cross validation 400:400
def run():
	clf = GradientBoostingClassifier(learning_rate=0.3, n_estimators=110, max_depth=3)
	List_x = [None] * 8
	List_y = [None] * 8

	x_name = str("dataSets/dataset1_X.npy")
	y_name = str("dataSets/dataset1_Y.npy")

	for i in range(8):
		List_x[i] = np.load(x_name)
		List_y[i] = np.load(y_name)

		x_name = "dataSets/dataset" + str(i + 2) + "_X.npy"
		y_name = "dataSets/dataset" + str(i + 2) + "_Y.npy"

	count = 0

	a = [0,1,2,3,4,5,6,7]
	comb = combinations([0,1,2,3,4,5,6,7], 4)


	for i in list(comb):

		x_train = np.zeros((0, 6))
		y_train = np.zeros((0))

		x_test = np.zeros((0, 6))
		y_test = np.zeros((0))

		setOfArray = list(i)
		#print setOfArray
		for j in setOfArray:
			x_train =  np.concatenate((x_train, List_x[j]), axis = 0)
			y_train = np.concatenate((y_train, List_y[j]), axis = None)
		#print x_train
		complementArray = list(set(a) - set(i))
		#print complementArray
		for p in complementArray:
			x_test = np.concatenate((x_test, List_x[int(p)]), axis = 0)
			y_test = np.concatenate((y_test, List_y[int(p)]), axis = None)

		clf.fit(x_train, y_train)
		result = clf.predict(x_test)
		
		for j in range(400):
			if y_test[j] == result[j]:
				count = count + 1
#	print count 
	print(count/28000.0)

if __name__ == '__main__':
	run()