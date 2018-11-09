import numpy as np
from sklearn import svm

def test_cross(clf):
    ##clf = svm.SVC(gamma = 'scale')

    List_x = [None] * 8
    List_y = [None] * 8

    x_name = str("dataSets\\dataset1_X.npy")
    y_name = str("dataSets\\dataset1_Y.npy")

    for i in range(8):
        List_x[i] = np.load(x_name)
        List_y[i] = np.load(y_name)

        x_name = "dataSets\\dataset" + str(i + 2) + "_X.npy"
        y_name = "dataSets\\dataset" + str(i + 2) + "_Y.npy"

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
    
    return (count/800)