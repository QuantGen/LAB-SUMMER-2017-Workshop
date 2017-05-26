from __future__ import print_function
import sys
import numpy as np
import pandas as pd     # read csv as data frame
import matplotlib.pyplot as plt
import math
import os

PATH_ROOT = '/mnt/research/quantgen/projects/demo/maize'
os.chdir(PATH_ROOT+"/code")

from fit_NN import fit_NN
nepochs = 150

# Read data
X = pd.read_csv(PATH_ROOT + '/data/Img_Data.csv')
Y = pd.read_csv(PATH_ROOT + '/data/Pheno_Data.csv')

sets = Y["ID"].as_matrix()
folds = np.unique(sets)
n = Y.shape[0]

yHatCV = np.zeros(n)
corFold = np.zeros(folds.shape[0])
X = X.as_matrix()
y = Y["Y"].as_matrix().astype(float)
y = y-np.mean(y)

# Matrices to save loss value
lossTRN = pd.DataFrame(index=range(nepochs),columns=folds)
lossTST = pd.DataFrame(index=range(nepochs),columns=folds)
corTST = pd.DataFrame(index=range(nepochs),columns=folds)

# Define layers for model
inlayer = {'nodes':50,'lambda2':0.05,'dropout':0.3}
hiddenlayers = [{'nodes':40,'lambda2':0.005,'dropout':0.3}]	

np.random.seed(123) 
for i in range(0,folds.shape[0]):
	partition = np.zeros(n)
	indexNA = sets==folds[i]
	partition[indexNA] = 1
	fm = fit_NN(X=X,y=y,partition=partition,inlayer=inlayer,hiddenlayers=hiddenlayers,nepochs=nepochs)
	
	# prediction
	yHat = fm['yHat'][indexNA]
	corre = fm['corTST']
	print("------- Correlation y & yHat: ", corre)
	corFold[i] = corre
	yHatCV[indexNA] = yHat
	lossTRN[folds[i]][:] = fm['LossTRN']
	lossTST[folds[i]][:] = fm['LossVAL']
	corTST[folds[i]][:] = fm['accuracy']

corOVERALL = np.corrcoef(y, yHatCV)[0, 1]
print("Overall Correlation: ", corOVERALL)

pd.DataFrame({'Trial':folds,'Correlation':corFold})

plt.scatter(yHatCV,y)
plt.show()

corTST.plot()
plt.show()

lossTRN.plot()
plt.show()

lossTST.plot()
plt.show()

tmp = pd.DataFrame({'TRN':np.mean(lossTRN,1),'TST':np.mean(lossTST,1)})
tmp.plot()
plt.show()
