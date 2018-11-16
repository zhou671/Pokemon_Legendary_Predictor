import pandas as pd 
import numpy as np 

def generateData(data_pos, data_neg, filename):
    n, d = data_pos.shape
    d = d - 1
    for i in range(8):
        namex = filename + "X" + str(i) + ".npy"
        namey = filename + "Y" + str(i) + ".npy"
        x = np.zeros((100,d))
        y = np.zeros((100))

        if i < 7:
            x[0:92] = data_pos[(i*92):(i*92+92), 0:d]
            y[0:92] = data_pos[(i*92):(i*92+92), d]
            x[92:100] = data_neg[(i*8):(i*8+8), 0:d]
            y[92:100] = data_neg[(i*8):(i*8+8), d]

        else:
            x[0:91] = data_pos[644:735, 0:d]
            y[0:91] = data_pos[644:735, d]
            x[91:100] = data_neg[56:65, 0:d]
            y[91:100] = data_neg[56:65, d]

        np.save(namex,x)
        np.save(namey,y)
    


if __name__ == '__main__':

    raw_csv = pd.read_csv("Pokemon.csv")
    type2 = raw_csv['Type 2']
    type1 = raw_csv['Type 1']
    raw_csv = raw_csv.drop(['#','Type 1', 'Type 2','Total','Generation','Name'], axis = 1)
    
    unq_type2 = type2.unique()
    dic_type = {unq_type2[i] : i for i in range(19)}
    cleaned_type = np.zeros((800,19))
    for i in range(800):
        cleaned_type[i,dic_type[type1[i]]] = 2
        cleaned_type[i,dic_type[type2[i]]] = 1

    data = raw_csv.values
    data = np.concatenate((cleaned_type, data),axis = 1)
    true_columns = list(np.where(data[:,25] == True))
    false_columns = list(np.where(data[:,25] == False))
    data[true_columns, 25] = -1
    data[false_columns, 25] = 1
    data_pos = data[np.where(data[:,25] == 1)]
    data_neg = data[np.where(data[:,25] == -1)]

    np.random.shuffle(data_pos)
    np.random.shuffle(data_pos)
    np.random.shuffle(data_neg)
    np.random.shuffle(data_neg)

    np.save("dataWtype//data.npy", data)
    np.save("dataWOtype//data.npy", data)

    generateData(data_pos, data_neg, "dataWtype//")
    generateData(data_pos[:, 19:], data_neg[:, 19:], "dataWOtype//")