# preprocess.py_withoutType

#import necessary librarys
import pandas as pd 
import numpy as np 

def run():
    #read csv and remove columns for types
    raw_csv = pd.read_csv("Pokemon.csv")
    raw_csv = raw_csv.drop(columns = ['Type 1', 'Type 2','Total','Generation','Name'])

    #convert csv to numpy array
    data = raw_csv.values

    #obtain all the index for true & false in columns 10
    true_columns = list(np.where(data[:,7] == True))
    false_columns = list(np.where(data[:,7] == False))

    #change true -> 0 and false -> 1
    data[true_columns, 7] = -1
    data[false_columns, 7] = 1

    #seprate data into 2 subsets
    data_pos = data[np.where(data[:,7] == 1)]
    data_neg = data[np.where(data[:,7] == -1)]

    #shuffle both subsets
    np.random.shuffle(data_pos)
    np.random.shuffle(data_neg)
    #print(data_pos)
    #seprate subdatas into 8 subsets
    
    for i in range(8):
        namex = "11.14_data//X" + str(i) + ".npy"
        namey = "11.14_data//Y" + str(i) + ".npy"
        x = np.zeros((100,7))
        y = np.zeros((100))

        if i < 7:
            x[0:92] = data_pos[(i*92):(i*92+92), 0:7]
            y[0:92] = data_pos[(i*92):(i*92+92), 7]
            x[92:100] = data_neg[(i*8):(i*8+8), 0:7]
            y[92:100] = data_neg[(i*8):(i*8+8), 7]

        else:
            x[0:91] = data_pos[644:735, 0:7]
            y[0:91] = data_pos[644:735, 7]
            x[91:100] = data_neg[56:65, 0:7]
            y[91:100] = data_neg[56:65, 7]

        np.save(namex,x)
        np.save(namey,y)
    

if __name__ == '__main__':
    run()







