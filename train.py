import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import OneClassSVM 

# args
# whatever x
# whatever y
# modelType has to be 0,1,2
#   modelType = 0 : gradient
#   modelType = 1 : LinearSVM
#   modelType = 2 : 2ndKernelSVM

class train:
    def __init__(self, x, y, modelType):
        self.C = np.array(range(10) / 10.0)
        self.rate = np.array(range(5)) / 100.0 + 0.01
        self.n = np.array(range(10)) * 10.0 + 150
        self.model = None
        self.x = [x, x[37:44]]
        # self.x_w = x
        # self.x_wo = x[37:44,:]
        self.y = y
        self.modelType = modelType
        self.bs_acc = 0

    def booststrapping(self, x, model):
        n = len(x)
        acc = np.zeros(30)
        for n in range(30):
            train_samples = list(np.random.randint(0,n,n))
            test_samples = list(set(range(n)) - set(train_samples))
            model.fit(x[train_samples], y[train_samples])
            acc[b] = np.mean(y[test_samples] == model.predict[x[test_samples]])
        return np.mean(acc)

    def training(self):
        paraFunc = self.modelfuncs[self.modelType]
        paraFunc(self)
        return self.model
       
    def gradientPara(self):
        bs_acc = 0
        for i in range(2):
            for j in range(5):
                for k in range(10):
                    model = GradientBoostingClassifier(learning_rate = self.rate[j], n_estimators = self.n[k], max_depth=3)
                    acc = self.booststrapping(self, x[i], model)
                    if acc > bs_acc:
                        bs_acc = acc
                        self.model = model
        
    
    def linearPara(self):
        bs_acc = 0
        for i in range(2):
            for j in range(10):
                model = svm.LinearSVC(penalty = 'l2', loss = 'squared_hinge', dual = False, C = self.C[j])
                acc = self.booststrapping(self, x[i], model)
                if acc > bs_acc:
                        bs_acc = acc
                        self.model = model

    def kerPara(self):
        bs_acc = 0
        for i in range(2):
            for j in range(10):
                model = svm.SVC(kernel = 'poly', degree = 2, C = 0.5, gamma = 'scale')
                acc = self.booststrapping(self, x[i], model)
                if acc > bs_acc:
                        bs_acc = acc
                        self.model = model

    modelfuncs = {gradientPara, linearPara, kerPara}