#!/usr/bin/python

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics, model_selection, linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = "usina72.csv"
tracefile = "regressao.csv"
kneighbors = 3
jobs = 90


def debug(text):
	print(str(text));

def saveresults(regressor, f3_mse, f3_var, f5_mse, f5_var):
	div_str = ","
	text_file = open(tracefile, "a")
	txt = regressor + div_str + str(f3_mse) + div_str + str(f3_var) + div_str + str(f5_mse) + div_str + str(f5_var)
	text_file.write(txt+'\n')
	text_file.close()

#======================
def LinearRegr(X_train, X_val, y_train, y_val):
	regr = LinearRegression()
	y_pred = regr.fit(X_train, y_train).predict(X_val)
	error = {y_val, y_pred - y_val}
	np.savetxt('lr.csv', error, delimiter=',', header='error', comments='')
	mse =metrics.mean_squared_error(y_val, y_pred)
	var = metrics.explained_variance_score(y_val, y_pred, multioutput='variance_weighted')
	print mse
	print var
	return (mse, var)

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
	knn = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='manhattan', n_jobs=jobs)
	y_pred = knn.fit(X_train,y_train).predict(X_test)
	mse =metrics.mean_squared_error(y_test, y_pred)
	var = metrics.r2_score(y_test, y_pred)
	return(mse, var)

#=======================
def MlpRegr(X_train, X_test, y_train, y_test):
	debug("Calculando MLP")
	regr = MLPRegressor(hidden_layer_sizes=(5,), activation='relu', solver='adam', learning_rate='adaptive', max_iter=1000, learning_rate_init=0.01, alpha=0.01)
	
	y_pred = regr.fit(X_train, y_train).predict(X_test)
	debug("MLP: " + str(metrics.mean_absolute_error(y_test, y_pred)))

#=======================
def DsfRegr(X_train, X_test, y_train, y_test):
	regr = DecisionTreeRegressor(max_depth=8, splitter='best')
	y_pred = regr.fit(X_train, y_train).predict(X_test)
	mse =metrics.mean_squared_error(y_test, y_pred)
	var = metrics.r2_score(y_test, y_pred)
	return(mse, var)


#=======================
def RandForestRegr(X_train, X_test, y_train, y_test):
	regr = RandomForestRegressor(max_depth=2, n_estimators=100, n_jobs=jobs)
	y_pred = regr.fit(X_train, y_train).predict(X_test)
	mse =metrics.mean_squared_error(y_test, y_pred)
	var = metrics.r2_score(y_test, y_pred)
	return(mse, var)


#=======================
def GradBoostRegr(X_train, X_test, y_train, y_test):
	regr = GradientBoostingRegressor(n_estimators=100, max_features='log2', min_samples_split=2, max_depth=1, learning_rate=0.01, loss='quantile', criterion='mse')
	y_pred = regr.fit(X_train, y_train).predict(X_test)
	mse =metrics.mean_squared_error(y_test, y_pred)
	var = metrics.r2_score(y_test, y_pred)

#=======================
def main():
	debug("Carregando dados")
	#carrega dados e divide
	
	dados = pd.read_csv(data)
	y1 = dados['f3'].values.reshape(-1,1)
	y2 = dados['f4'].values.reshape(-1,1)

	#f4, f5,f6,f9,f10,f11,f12  dsds
	X = dados[['f5','f6']].values

	
	X_t, X_test, y1_t, y1_test, y2_t, y2_test = train_test_split(X, y1, y2, test_size=0.5, random_state=48)
	X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(X_t, y1_t, y2_t, test_size=0.3, random_state=48)
	#criar vetores de teste, mudar teste para validacao
	#nao usar cross validation
	#
	
	#normalizando dados
	min_max_scaler = preprocessing.MinMaxScaler()
	X_train = min_max_scaler.fit_transform(X_train)
	X_test = min_max_scaler.fit_transform(X_test)
	X_val = min_max_scaler.fit_transform(X_val)
	
	y1_test = min_max_scaler.fit_transform(y1_test)
	y1_train = min_max_scaler.fit_transform(y1_train)
	y1_val = min_max_scaler.fit_transform(y1_val)
	
	y2_test = min_max_scaler.fit_transform(y2_test)
	y2_train = min_max_scaler.fit_transform(y2_train)
	y2_val = min_max_scaler.fit_transform(y2_val)
	
	
	(mse1, var1) = LinearRegr(X_train, X_val, y1_train, y1_val)
	#(mse2, var2) = LinearRegr(X_train_minmax, X_test_minmax, y2_train, y2_test)
	#saveresults("Linear Regression", mse1, var1, mse2, var2)
	#(mse1, var1) = knnRegr(X_train_minmax, X_test_minmax, y1_train, y1_test)
	#(mse2, var2) = knnRegr(X_train_minmax, X_test_minmax, y2_train, y2_test)
	#saveresults("KNN", mse1, var1, mse2, var2)
	#(mse1, var1) = DsfRegr(X_train_minmax, X_test_minmax, y2_train.reshape(-1,), y2_test.reshape(-1,))
	#(mse2, var2) = DsfRegr(X_train_minmax, X_test_minmax, y1_train.reshape(-1,), y1_test.reshape(-1,))
	#saveresults("Decision Tree", mse1, var1, mse2, var2)
	#(mse1, var1) = RandForestRegr(X_train_minmax, X_test_minmax, y1_train.reshape(-1,), y1_test.reshape(-1,))
	#(mse1, var1) = RandForestRegr(X_train_minmax, X_test_minmax, y2_train.reshape(-1,), y2_test.reshape(-1,))
	#saveresults("Random Forest", mse1, var1, mse2, var2)
	#GradBoostRegr(X_train_minmax, X_test_minmax, y1_train.reshape(-1,), y1_test.reshape(-1,))


	#MlpRegr(X_train_minmax, X_test_minmax, y1_train.reshape(-1,), y1_test.reshape(-1,))
	#MlpRegr(X_train_minmax, X_test_minmax, y2_train.reshape(-1,), y2_test.reshape(-1,))
	#GradBoostRegr(X_train_minmax, X_test_minmax, y1_train.reshape(-1,), y1_test.reshape(-1,))
	#GradBoostRegr(X_train_minmax, X_test_minmax, y2_train.reshape(-1,), y2_test.reshape(-1,))
	#SvrRegr(X_train_minmax, X_test_minmax, y1_train.reshape(-1,), y1_test.reshape(-1,))
	#SvrRegr(X_train_minmax, X_test_minmax, y2_train.reshape(-1,), y2_test.reshape(-1,))


if __name__ == "__main__":
	main()
