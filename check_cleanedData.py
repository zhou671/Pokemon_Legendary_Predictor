import numpy as np 

def check():

    List_x = [None] * 8
    for i in range(8):
        x_name = "11.14_data//X" + str(i) + ".npy"
        List_x[i] = np.load(x_name)

    x = np.zeros((800, 7))
    for i in range(8):
        x[i*100:i*100+100] = List_x[i]
    #x = np.unique(x)
    print(x.size/7)


if __name__ == '__main__':
    check()