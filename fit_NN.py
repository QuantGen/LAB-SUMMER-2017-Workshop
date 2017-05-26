from __future__ import print_function
import numpy as np
import math
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten 
from keras import regularizers
from keras.constraints import maxnorm
import keras.backend as K

#-----------------------------------------------
def correlation(y_true, y_pred):
#-----------------------------------------------
	y_true = K.cast(y_true, dtype='float32')
	y_pred = K.cast(y_pred, dtype='float32')
	x11 = y_true-K.mean(y_true, axis = -1)
	x22 = y_pred-K.mean(y_pred, axis = -1)
	a1=K.tf.reduce_sum(K.tf.multiply(x11,x22))
	a2=K.tf.reduce_sum(K.tf.multiply(x11,x11))
	a3=K.tf.reduce_sum(K.tf.multiply(x22,x22))
	corre = a1/(K.sqrt(a2)*K.sqrt(a3))
	return corre
#

#-----------------------------------------------
def fit_NN(X,y,partition,inlayer=[],hiddenlayers=[],outlayer=[],nepochs=20,loss="mean_squared_error",optimizer="adam",batch_size=None):
#-----------------------------------------------
	#np.random.seed(seed)  # for reproducibility
	n = X.shape[0]
	p = X.shape[1]
	indexNA = partition==1
	nTST = np.sum(indexNA)
	nTRN = np.sum(-indexNA)
	yHat = np.zeros(n)
	accuValue = np.zeros(nepochs)
	valueLoss = np.zeros(nepochs)
	valueLossVal = np.zeros(nepochs)
	
	print("Data were splitted in TRN(",nTRN,") samples and TST(",nTST,") samples.")
	
	xTRN=X[-indexNA,:]
	yTRN=y[-indexNA]
	xTST=X[indexNA,:]
	yTST=y[indexNA]
	batch_size = np.max([nTRN,nTST]) if batch_size is None else batch_size
	
	# Model definition 
	model = Sequential()
	kernel0 = 'normal' if not 'kernel0' in list(inlayer) else inlayer['kernel0']
	lambda1 = 0 if not 'lambda1' in list(inlayer) else inlayer['lambda1']
	lambda2 = 0 if not 'lambda2' in list(inlayer) else inlayer['lambda2']
	activation = 'relu' if not 'activation' in list(inlayer) else inlayer['activation']
	model.add(Dense(inlayer['nodes'], input_dim=p, kernel_initializer=kernel0,W_regularizer=regularizers.l1_l2(lambda1,lambda2),activation=activation))
	if 'dropout' in list(inlayer):
		model.add(Dropout(inlayer['dropout']))
		print("==== Input layer. Nodes=",inlayer['nodes'],". Kernel_initializer=",kernel0,". L1=",lambda1,". L2=",lambda2,". Activation=",activation,". Dropout=",inlayer['dropout'])
	else:	print("==== Input layer. Nodes=",inlayer['nodes'],". Kernel_initializer=",kernel0,". L1=",lambda1,". L2=",lambda2,". Activation=",activation)
		
	for j in range(0,len(hiddenlayers)):
		kernel0 = 'normal' if not 'kernel0' in list(hiddenlayers[0]) else hiddenlayers[0]['kernel0']
		lambda1 = 0 if not 'lambda1' in list(hiddenlayers[j]) else hiddenlayers[j]['lambda1']
		lambda2 = 0 if not 'lambda2' in list(hiddenlayers[j]) else hiddenlayers[j]['lambda2']
		model.add(Dense(hiddenlayers[j]['nodes'], kernel_initializer=kernel0,W_regularizer=regularizers.l1_l2(lambda1,lambda2)))
		if 'dropout' in list(hiddenlayers[j]):
			model.add(Dropout(hiddenlayers[j]['dropout']))
			print("==== Internal layer ",str(j+1),". Nodes=",hiddenlayers[j]['nodes'],". Kernel_initializer=",kernel0,". L1=",lambda1,". L2=",lambda2,". Dropout=",hiddenlayers[j]['dropout'])
		else:	print("==== Internal layer ",str(j+1),". Nodes=",hiddenlayers[j]['nodes'],". Kernel_initializer=",kernel0,". L1=",lambda1,". L2=",lambda2)
	
	kernel0 = 'normal' if not 'kernel0' in list(outlayer) else outlayer['kernel0']	
	print("==== Output layer. Kernel_initializer=",kernel0)
	model.add(Dense(1, kernel_initializer=kernel0))
	
	# Compile model
	model.compile(loss=loss, optimizer=optimizer,metrics=[correlation])
		
	history = model.fit(xTRN, yTRN,batch_size=batch_size,verbose=1,epochs=nepochs,validation_data=(xTST, yTST))
	
	# prediction
	yHatTST = model.predict(xTST, batch_size=batch_size,verbose=0)[:,0]
	correTST = np.corrcoef(yTST, yHatTST)[0, 1]
	yHatTRN = model.predict(xTRN, batch_size=batch_size,verbose=0)[:,0]
	correTRN = np.corrcoef(yTRN, yHatTRN)[0, 1]
		
	accuValue = history.history['val_correlation']
	valueLossVal = history.history['val_loss']
	valueLoss = history.history['loss']
			
	print("Correlation TRN(y,yHat): ",correTRN)
	print("Correlation TST(y,yHat): ",correTST)
	yHat[indexNA] = yHatTST
	yHat[-indexNA] = yHatTRN
	
	out = {'yHat':yHat,'accuracy':accuValue,'LossTRN':valueLoss,'LossVAL':valueLossVal,'corTST':correTST}
	return(out)
#

print("Function fit_NN loaded")
