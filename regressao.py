#!/usr/bin/python

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
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
def LinearRegr(X_train, X_test, y_train, y_test):
	regr = LinearRegression()
	y_pred = regr.fit(X_train, y_train).predict(X_test)
	mse =metrics.mean_squared_error(y_test, y_pred)
	var = metrics.r2_score(y_test, y_pred)
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
def baseline_model():
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def MlpRegr(X_train, X_test, y_train, y_test):
	debug("Calculando MLP")
	seed = 7
	np.random.seed(seed)
	estimators = []
	estimators.append(('standardize', StandardScaler()))
	estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
	pipeline = Pipeline(estimators)
	kfold = KFold(n_splits=10, random_state=seed)
	results = cross_val_score(pipeline, X, Y, cv=kfold)
	print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#debug("MLP: " + str(metrics.mean_absolute_error(y_test, pipeline)))

#=======================
def DsfRegr(X_train, X_test, y_train, y_test):
	regr = DecisionTreeRegressor(max_depth=8, splitter='best')
	y_pred = regr.fit(X_train, y_train).predict(X_test)
	mse =metrics.mean_squared_error(y_test, y_pred)
	var = metrics.r2_score(y_test, y_pred)
	return(mse, var)


#=======================
def RandForestRegr(X_train, X_test, y_train, y_test):
	regr = RandomForestRegressor(max_depth=8, n_estimators=71, n_jobs=jobs)
	y_pred = regr.fit(X_train, y_train).predict(X_test)
	mse =metrics.mean_squared_error(y_test, y_pred)
	var = metrics.r2_score(y_test, y_pred)
	return(mse, var)


#=======================
def GradBoostRegr(X_train, X_test, y_train, y_test):
	debug("Calculando Gradient Boosting")
	loss = ['ls', 'lad', 'huber', 'quantile']
	min_mse = 9999
	min_param = ""
	for es in range(1,4):
		es = es * 50
		for d in range(1,10):
			for s in range(2,8):
				for l in loss:
					regr = GradientBoostingRegressor(n_estimators=es, max_depth=d, min_samples_split=s, learning_rate=0.01, loss=l, criterion='mse')
					y_pred = regr.fit(X_train, y_train).predict(X_test)
					mse =metrics.mean_squared_error(y_test, y_pred)
					var = metrics.r2_score(y_test, y_pred)
					debug("Gradient Boosting: ("+str(es)+","+str(d)+","+str(s)+","+str(l)+")" + str(mse))
					if mse < min_mse:
						min_mse = mse
						min_param = "("+str(es)+","+str(d)+","+str(s)+","+str(l)+")"
	print "Valores otimos: "+min_param+ " mse: "+ str(min_mse)

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

	#(mse1, var1) = LinearRegr(X_train_minmax, X_test_minmax, y1_train, y1_test)
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


	MlpRegr(X_train_minmax, X_test_minmax, y1_train.reshape(-1,), y1_test.reshape(-1,))
	#MlpRegr(X_train_minmax, X_test_minmax, y2_train.reshape(-1,), y2_test.reshape(-1,))
	#GradBoostRegr(X_train_minmax, X_test_minmax, y1_train.reshape(-1,), y1_test.reshape(-1,))
	#GradBoostRegr(X_train_minmax, X_test_minmax, y2_train.reshape(-1,), y2_test.reshape(-1,))
	#SvrRegr(X_train_minmax, X_test_minmax, y1_train.reshape(-1,), y1_test.reshape(-1,))
	#SvrRegr(X_train_minmax, X_test_minmax, y2_train.reshape(-1,), y2_test.reshape(-1,))


if __name__ == "__main__":
	main()
