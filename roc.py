import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def run(y_real, y_pred1, y_pred2, y_pred3, k):
	
	#code for prototype, will be delete when real testing
	# y_real = np.random.choice(2, 800, p = [0.92, 0.08])
	# y_pred1 = np.random.randint(1, size = 800)
	# y_pred2 = np.random.choice(2, 800, p = [0.92, 0.08])
	# y_pred3 = np.random.choice(2, 800, p = [0.92, 0.08])

	#generate results for 1st set
	tn1,fp1,fn1,tp1 = confusion_matrix(y_real, y_pred1).ravel()
	specificity1 = float(tn1) / (tn1+fp1)
	sensitivity1 = float(tp1) / (tp1+fn1)
	plt.plot(specificity1,sensitivity1, marker = 'o', color = 'r', label = 'Linear')
	#generate results for 2nd set
	tn2,fp2,fn2,tp2 = confusion_matrix(y_real, y_pred2).ravel()
	specificity2 = float(tn2) / (tn2+fp2)
	sensitivity2 = float(tp2) / (tp2+fn2)
	plt.plot(specificity2,sensitivity2, marker = 'o', color = 'g', label = '2ndSVM')
	
	#generate results for 3rd set
	tn3,fp3,fn3,tp3 = confusion_matrix(y_real, y_pred3).ravel()
	specificity3 = float(tn3) / (tn3+fp3)
	sensitivity3 = float(tp3) / (tp3+fn3)
	plt.plot(specificity3,sensitivity3, marker = 'o', color = 'b', label = 'Gradient')
	
	#labels and title setting
	plt.title('ROC')
	plt.xlabel("Specificity")
	plt.ylabel("Sensitivity")
	plt.legend(loc='lower left')

	name = ["8-fold.png", "2-fold.png"]
	#graph saving as .png file
	plt.savefig("./Results/" + name[k])
	
