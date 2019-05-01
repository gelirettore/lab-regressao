#!/usr/bin/python

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = "usina72.csv"
kneighbors = 3

def debug(text):
	print(str(text));

#======================
def LinearRegr(X_train, X_test, y_train, y_test):
	debug("Calculando regressao linear")
	#Cria modelo
	regr = LinearRegression()
	
	#Treina modelo
	regr.fit(X_train, y_train)
	y_pred = regr.predict(X_test)
	debug('Regressao Linear (MSE):' + str(metrics.mean_absolute_error(y_test, y_pred)))

#=======================
def SvrRegr(X_train, X_test, y_train, y_test):
	debug("Calculando SVR")
	#SVR + RBF
	svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
	svr_lin = SVR(kernel='linear', C=100, gamma='auto')
	svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)
	
	debug("Fitting and predicting...")
	y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
	y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
	y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
	
	debug("SVR(rbf): " + str(metrics.mean_absolute_error(y_test, y_rbf)))
	debug("SVR(linear): " + str(metrics.mean_absolute_error(y_test, y_lin)))
	debug("SVR(poly): " + str(metrics.mean_absolute_error(y_test, y_poly)))
#=======================
def knnRegr(X_train, X_test, y_train, y_test):
	debug("Calculando knn")
	knn = KNeighborsRegressor(kneighbors)
	y_pred = knn.fit(X_train,y_train).predict(X_test)
	debug("KNN(3): " + str(metrics.mean_absolute_error(y_test, y_pred)))

#=======================
def MlpRegr(X_train, X_test, y_train, y_test):
	debug("Calculando MLP")
	mlp = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
					   learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
					   random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
					   nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
					   epsilon=1e-08)
	y_pred = mlp.fit(X_train, y_train).predict(X_test)
	debug("MLP: " + str(metrics.mean_absolute_error(y_test, y_pred)))


#=======================
def main():
	debug("Carregando dados")
	#carrega dados e divide

	dados = pd.read_csv(data)
	y1 = dados['f3'].values.reshape(-1,1)
	y2 = dados['f5'].values.reshape(-1,1)

	
	X = dados[['f4','f6','f9','f10','f11']].values
	
	X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.5, random_state=48)
	
	#normalizando dados
	min_max_scaler = preprocessing.MinMaxScaler()
	X_train_minmax = min_max_scaler.fit_transform(X_train)
	X_test_minmax = min_max_scaler.fit_transform(X_test)
			
	y1_test_minmax = min_max_scaler.fit_transform(y1_test)
	y1_train_minmax = min_max_scaler.fit_transform(y1_train)

	y2_test_minmax = min_max_scaler.fit_transform(y2_test)
	y2_train_minmax = min_max_scaler.fit_transform(y2_train)

	LinearRegr(X_train_minmax, X_test_minmax, y1_train, y1_test)
	LinearRegr(X_train_minmax, X_test_minmax, y2_train, y2_test)
	#SvrRegr(X_train_minmax, X_test_minmax, y1_train.reshape(-1,), y1_test.reshape(-1,))
	#SvrRegr(X_train_minmax, X_test_minmax, y2_train.reshape(-1,), y2_test.reshape(-1,))
	knnRegr(X_train_minmax, X_test_minmax, y1_train, y1_test)
	knnRegr(X_train_minmax, X_test_minmax, y2_train, y2_test)
	MlpRegr(X_train_minmax, X_test_minmax, y2_train, y2_test)
	

if __name__ == "__main__":
	main()
