#!/usr/bin/python

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


data = "usina72.csv"

def debug(text):
    print(str(text));

def LinearRegr(X_train, X_test, y_train, y_test):
    #Cria modelo
    regr = LinearRegression()
    
    #Treina modelo
    regr.fit(X_train, y_train)
        
    score_train = regr.score(X_train, y_train)
    score_test = regr.score(X_test, y_test)
    debug('Regressao Linear (treino): %.2f' % score_train)
    debug('Regressao Linear (teste): %.2f' % score_test)

def SvrRegr(X_train, X_test, y_train, y_test):
    #SVR + RBF
    clf_rbf = SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1)
    clf_rbf.fit(X_train, y_train)
    score_rbf = clf_rbf.score(X_test, y_test)
    predict_rbf = clf_rbf.predict(X_test)
    print score_rbf
    debug("SVR + RBF score: " + score_rbf)

def main():
    debug("Carregando dados")
    #carrega dados e divide
    
    dados = pd.read_csv(data)
    y1 = dados['f3']
    y2 = dados['f5']
    y  = np.concatenate((y1, y2), axis=1)
    print y

    x1 = dados['f4']
    X2 = dados['f6']
    x3 = dados['f9']
    x4 = dados['f10']
    x5 = dados['f11']
#	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=50)
#	min_max_scaler = preprocessing.MinMaxScaler()
#	X_train_minmax = min_max_scaler.fit_transform(X_train)
#	X_test_minmax = min_max_scaler.fit_transform(X_test)

#LinearRegr(X_train_minmax, X_test_minmax, y_train, y_test)
#SvrRegr(X_train_minmax, X_test_minmax, y_train, y_test)


if __name__ == "__main__":
    main()
