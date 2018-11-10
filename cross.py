import numpy as np
from itertools import combinations
from sklearn import svm

def run():
    clf = svm.SVC(gamma = 'scale')

    ## read X0-X7, Y0-Y7
    List_x = [None] * 8
    List_y = [None] * 8
    for i in range(8):
        x_name = "data\\X" + str(i) + ".npy"
        y_name = "data\\Y" + str(i) + ".npy"
        List_x[i] = np.load(x_name)
        List_y[i] = np.load(y_name)

    ## training acc
    count = 0
    x = np.zeros((800, 6))
    y = np.zeros((800))
    for i in range(8):
        x[i*100:i*100+100] = List_x[i]
        y[i*100:i*100+100] = List_y[i]

    clf.fit(x,y)
    result = clf.predict(x)
    count = len(y[np.where(y == result)])
    print("Training acc: " + str(count / 800))

    ## 7-1 test acc
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
    
    print("7-1 test acc: " + str(count / 800))

    ## 4-4 test acc
    count = 0
    comb = list(combinations(range(8), 4))
    for i in range(70):
        x = np.zeros((400,6))
        y = np.zeros((400))
        s = set(range(8))
        for j in range(4):
            x[j*100:j*100+100] = List_x[comb[i][j]]
            y[j*100:j*100+100] = List_y[comb[i][j]]
            s.remove(comb[i][j])

        test_x = np.zeros((400,6))
        test_y = np.zeros((400))
        s = list(s)
        for j in range(4):
            test_x[j*100:j*100+100] = List_x[s[j]]
            test_y[j*100:j*100+100] = List_y[s[j]]

        clf.fit(x,y)
        result = clf.predict(test_x)
        count += len(y[np.where(test_y == result)])

    print("4-4 test acc: " + str(count / 28000))
    return

if __name__ == '__main__':
	run() 