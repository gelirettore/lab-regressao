#!/usr/bin/python

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = "usina72.csv"

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
	clf_rbf = SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1)
	clf_rbf.fit(X_train, y_train)
	
	#y_pred = clf_rbf.predict(X_test)
	#print "SVR(rbf)" + str(metrics.mean_absolute_error(y_test, y_pred))

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
	SvrRegr(X_train_minmax, X_test_minmax, y1_train.reshape(-1,), y1_test.reshape(-1,))


if __name__ == "__main__":
	main()
