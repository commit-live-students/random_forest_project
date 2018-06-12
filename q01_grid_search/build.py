import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

loan_data = pd.read_csv('data/loan_prediction.csv')
X = loan_data.iloc[:, :-1]
y = loan_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)

param_grid1 = {'max_features': ['sqrt', 4, 'log2'],
              'n_estimators': [10, 50, 120],
              'max_depth': [40, 20, 10],
              'max_leaf_nodes': [5, 10, 2]}

rfc = RandomForestClassifier(oob_score=True ,random_state=9)

def grid_search(X_train1, y_train1,modelR,params,cv=3): 
    GSCV_rfc = GridSearchCV(estimator=modelR, param_grid=params, cv=cv)
    GSCV_rfc.fit(X_train1,y_train1)
    param_list = GSCV_rfc.cv_results_['params']
    score1 = GSCV_rfc.cv_results_['mean_test_score']
    return GSCV_rfc,param_list,score1


