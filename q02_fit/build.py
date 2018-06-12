import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

loan_data = pd.read_csv('data/loan_prediction.csv')
X = loan_data.iloc[:, :-1]
y = loan_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)

param_grid1 = {'max_features': ['sqrt', 4, 'log2'],
              'n_estimators': [10, 50, 120],
              'max_depth': [40, 20, 10],
              'max_leaf_nodes': [5, 10, 2]}

rfc = RandomForestClassifier(oob_score = True,random_state=9)

def grid_search(X_train1, y_train1,modelR,params,cv): 
    GSCV_rfc = GridSearchCV(estimator=modelR, param_grid=params, cv=3)
    GSCV_rfc.fit(X_train1,y_train1)
    param_list = GSCV_rfc.cv_results_['params']
    scoreA = GSCV_rfc.cv_results_['mean_test_score']
    return GSCV_rfc,param_list,scoreA

GSCV_rfc1,param_list1,score1 = grid_search(X_train, y_train,rfc,param_grid1,3)

model = GSCV_rfc1.best_estimator_
y_pred = model.fit(X_train, y_train).predict(X_test)

def fit(X_test,y_test):
    acc_score = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    c_report = classification_report(y_test, y_pred)
    return conf_matrix,c_report,acc_score


