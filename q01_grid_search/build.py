# %load q01_grid_search/build.py
# Default imports

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import numpy as np

loan_data = pd.read_csv('data/loan_prediction.csv')
X_bal = loan_data.iloc[:, :-1]
y_bal = loan_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.33, random_state=9)
param_grid = {"max_features": ['sqrt', 4, "log2"],
              "n_estimators": [10, 50, 120],
              "max_depth": [40, 20, 10],
              "max_leaf_nodes": [5, 10, 2]}
classifier = RandomForestClassifier(random_state=9,oob_score=True)

def grid_search(X_train, y_train, classifier,param_grid,cv=3):


    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid,cv=3)
    grid_search.fit(X_train,y_train)
    y_prediction = grid_search.predict(X_test)
    return grid_search, tuple(grid_search.cv_results_['params']), grid_search.cv_results_['mean_test_score']

#grid_search(X_train, y_train, classifier, param_grid)
