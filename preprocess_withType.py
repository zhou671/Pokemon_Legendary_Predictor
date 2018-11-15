# preprocess_withType.py

#import necessary librarys
import pandas as pd 
import numpy as np 

def run():
    #read csv and remove columns for types
    raw_csv = pd.read_csv("Pokemon.csv")
    raw_csv = raw_csv.drop(columns = ['#','Type 1', 'Type 2','Total','Generation','Name'])

    #convert csv to numpy array
    data = raw_csv.values

    #read types
    raw_csv = pd.read_csv("Pokemon.csv")
    #create dictionary
    type2 = raw_csv['Type 2']
    type1 = raw_csv['Type 1']
    unq_type2 = type2.unique()
    dic_type = {unq_type2[i] : i for i in range(19)}
    
    cleaned_type = np.zeros((800,19))
    for i in range(800):
        pos_1 = dic_type[type1[i]]
        pos_2 = dic_type[type2[i]]
        weight_1 = 2
        weight_2 = 1
        cleaned_type[i,pos_1] = weight_1
        cleaned_type[i,pos_2] = weight_2


    data = np.concatenate((cleaned_type, data),axis = 1)
    #print(data[:,25])
    #obtain all the index for true & false in columns 10
    true_columns = list(np.where(data[:,25] == True))
    false_columns = list(np.where(data[:,25] == False))

    #change true -> 0 and false -> 1
    data[true_columns, 25] = -1
    data[false_columns, 25] = 1

    #seprate data into 2 subsets
    data_pos = data[np.where(data[:,25] == 1)]
    data_neg = data[np.where(data[:,25] == -1)]
    print(data)
    #there are 26 columns now

    np.save("11.14_data_type//processed_type.npy", cleaned_type)

    np.save("11.14_data_type//processed_data.npy", data)

    
    for i in range(8):
        namex = "11.15_Type_data//X" + str(i) + ".npy"
        namey = "11.15_Type_data//Y" + str(i) + ".npy"
        x = np.zeros((100,25))
        y = np.zeros((100))

        if i < 7:
            x[0:92] = data_pos[(i*92):(i*92+92), 0:25]
            y[0:92] = data_pos[(i*92):(i*92+92), 25]
            x[92:100] = data_neg[(i*8):(i*8+8), 0:25]
            y[92:100] = data_neg[(i*8):(i*8+8), 25]

        else:
            x[0:91] = data_pos[644:735, 0:25]
            y[0:91] = data_pos[644:735, 25]
            x[91:100] = data_neg[56:65, 0:25]
            y[91:100] = data_neg[56:65, 25]

        np.save(namex,x)
        np.save(namey,y)
    

if __name__ == '__main__':
    run()



