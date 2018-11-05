# Pokemon Legendary Predictor
#Project Goal: To predict Pokemon legendary using svm

Files exists:

    Pokemon.csv:
        Raw data

    x.npy:
        Data after preprocess
        Use commend:
            x = np.load("npy")
        To take values in x.npy

        x[:, 0] : unique id
            #we might not need this for training/testing
        x[:, 1] : total of all stat
        x[:, 2:7] : specific stat for a pokemon e.g. speed, attack...
        x[:, 8] : pokemon generation 
            #we might not need this for training/testing
        x[:, 9] : legendary label, 0 : false, 1 : true 
            #we might not need this for training/testing

    output.png:
        A big picture how does the data look like
    
Script:

    Done:
        None

    To do:
        A preprocess file to seperate x into 8 different groups

        A cross validation test 4-4, 7-1

        A main file(maybe)
