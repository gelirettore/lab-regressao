#!/usr/bin/python

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics, model_selection, linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import sys


data = "usina72.csv"
tracefile = "regressao.csv"
kneighbors = 3
jobs = 90

dados = pd.read_csv(data)
y1 = dados['f3'].values.reshape(-1,1)
y2 = dados['f4'].values.reshape(-1,1)
X = dados[['f5','f6']].values

X_t, X_test, y1_t, y1_test, y2_t, y2_test = train_test_split(X, y1, y2, test_size=0.5, random_state=48)
X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(X_t, y1_t, y2_t, test_size=0.3, random_state=48)

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

def debug(text):
	print(str(text));

def saveresults(regressor, f3_mse, f3_var, f5_mse, f5_var):
	return;
	div_str = ","
	text_file = open(tracefile, "a")
	txt = regressor + div_str + str(f3_mse) + div_str + str(f3_var) + div_str + str(f5_mse) + div_str + str(f5_var)
	text_file.write(txt+'\n')
	text_file.close()

#====================== PREDICT ==================
def Predict(model, X_train, X_val, y_train, y_val, file):
	y_pred = model.fit(X_train,y_train).predict(X_val)
	error = y_pred - y_val
	np.savetxt(file + ".csv", error, delimiter=',', header='error', comments='')
	mse = metrics.mean_squared_error(y_val, y_pred)
	var = metrics.explained_variance_score(y_val, y_pred, multioutput='variance_weighted')
	return (mse, var)

#======================
def LinearRegr():
	regr = LinearRegression()
	(mse1, var1) = Predict(regr, X_train, X_val, y1_train, y1_val, 'lr1')
	(mse2, var2) = Predict(regr, X_train, X_val, y2_train, y2_val, 'lr2')
	saveresults("Linear Regression", mse1, var1, mse2, var2)
	debug("Linear ["+str(mse1)+","+ str(var1)+","+ str(mse2)+","+ str(var2)+"]")


#=======================
def SvrRegr():
	regr = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)
	(mse1, var1) = Predict(regr, X_train, X_val, y1_train.reshape(-1,), y1_val.reshape(-1,), 'svr1')
	(mse2, var2) = Predict(regr, X_train, X_val, y2_train.reshape(-1,), y2_val.reshape(-1,), 'svr2')
	debug("SVR ["+str(mse1)+","+ str(var1)+","+ str(mse2)+","+ str(var2)+"]")
	saveresults("SVR", mse1, var1, mse2, var2)


#=======================
def knnRegr():
	regr = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='manhattan', n_jobs=jobs)
	(mse1, var1) = Predict(regr, X_train, X_val, y1_train, y1_val, 'knn1')
	(mse2, var2) = Predict(regr, X_train, X_val, y2_train, y2_val, 'knn2')
	saveresults("KNN", mse1, var1, mse2, var2)
	debug("KNN ["+str(mse1)+","+ str(var1)+","+ str(mse2)+","+ str(var2)+"]")

#=======================
def MlpRegr():
	#regr = MLPRegressor(hidden_layer_sizes=(5,3), activation='relu', solver='adam', learning_rate='adaptive', max_iter=1000, learning_rate_init=0.01, alpha=0.01)
	regr = MLPRegressor(hidden_layer_sizes=(5,3), activation='logistic', solver='adam', learning_rate='invscaling', max_iter=1000, learning_rate_init=0.01, alpha=0.01)
	(mse1, var1) = Predict(regr, X_train, X_val, y1_train.reshape(-1,), y1_val.reshape(-1,), 'mlp1')
	(mse2, var2) = Predict(regr, X_train, X_val, y2_train.reshape(-1,), y2_val.reshape(-1,), 'mlp2')
	saveresults("MLP", mse1, var1, mse2, var2)
	debug("MLP ["+str(mse1)+","+ str(var1)+","+ str(mse2)+","+ str(var2)+"]")


#=======================
def DTRegr():
	regr = DecisionTreeRegressor(max_depth=3, splitter='best')
	(mse1, var1) = Predict(regr, X_train, X_val, y1_train.reshape(-1,), y1_val.reshape(-1,), 'dt1')
	(mse2, var2) = Predict(regr, X_train, X_val, y2_train.reshape(-1,), y2_val.reshape(-1,), 'dt2')
	saveresults("Decision Tree", mse1, var1, mse2, var2)
	debug("Decision Tree ["+str(mse1)+","+ str(var1)+","+ str(mse2)+","+ str(var2)+"]")


#=======================
def RandForestRegr(d, e):
	regr = RandomForestRegressor(max_depth=int(d), n_estimators=int(e), n_jobs=jobs)
	#regr = RandomForestRegressor(max_depth=8, n_estimators=70, n_jobs=jobs)
	(mse1, var1) = Predict(regr, X_train, X_val, y1_train.reshape(-1,), y1_val.reshape(-1,), 'rf1')
	(mse2, var2) = Predict(regr, X_train, X_val, y2_train.reshape(-1,), y2_val.reshape(-1,), 'rf2')
	saveresults("Random Forest", mse1, var1, mse2, var2)
	debug("Random Forest ["+str(mse1)+","+ str(var1)+","+ str(mse2)+","+ str(var2)+"]")


#=======================
def GradBoostRegr():
	#regr = GradientBoostingRegressor(n_estimators=150, min_samples_split=2, max_depth=1, learning_rate=0.01, loss='quantile', criterion='mse')
	regr = GradientBoostingRegressor(n_estimators=50, random_state=0)
	(mse1, var1) = Predict(regr, X_train, X_val, y1_train.reshape(-1,), y1_val.reshape(-1,), 'gb1')
	(mse2, var2) = Predict(regr, X_train, X_val, y2_train.reshape(-1,), y2_val.reshape(-1,), 'gb2')
	debug("Gradient Boosting ["+str(mse1)+","+ str(var1)+","+ str(mse2)+","+ str(var2)+"]")
	saveresults("Gradient Boosting", mse1, var1, mse2, var2)

#=======================
def main(option, arg1, arg2):

	if option == 'linear':
			LinearRegr()
	elif  option == 'knn':
			knnRegr()
	elif option == 'dt':
			DTRegr()
	elif option == 'rf':
			RandForestRegr(arg1, arg2)
	elif option == 'gb':
			GradBoostRegr()
	elif option == 'mlp':
			MlpRegr()
	elif option == 'svr':
			SvrRegr()
	elif option == 'all':
			LinearRegr()
			knnRegr()
			DTRegr()
			RandForestRegr()
			GradBoostRegr()
			MlpRegr()
			SvrRegr()
	else:
			sys.exit("Use: "+sys.argv[0]+" <linear|knn|dt|rf|gb|mlp|svr|all>")



if __name__ == "__main__":
	if len(sys.argv) < 2:
		sys.exit("Use: "+sys.argv[0]+" <linear|knn|dt|rf|gb|mlp|svr|all>")
	main(sys.argv[1], sys.argv[2], sys.argv[3])
