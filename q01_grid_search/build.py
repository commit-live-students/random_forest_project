# %load q01_grid_search/build.py
# Default imports

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

loan_data = pd.read_csv('data/loan_prediction.csv')
X_bal = loan_data.iloc[:, :-1]
y_bal = loan_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.33, random_state=9)
param_grid = {'max_features': ['sqrt', 4, 'log2'],
              'n_estimators': [10, 50, 120],
              'max_depth': [40, 20, 10],
              'max_leaf_nodes': [5, 10, 2]}

#acc_score = make_scorer(accuracy_score)
# Write your solution here :
clf = RandomForestClassifier(random_state=9,oob_score=True)
model = GridSearchCV(clf,param_grid,cv=3)
def grid_search(X_train,y_train,model,param_grid,cv=3):
    grid = GridSearchCV(estimator=clf,param_grid=param_grid,cv=3)
    grid.fit(X_train,y_train)
    #test_score = grid_obj.cv_results_['mean_test_score']
    #train_score = grid_obj.cv_results_['params']
    #param_values = sorted([str(x) for x in list(grid_obj.param_grid.items())[0][1]])
    #param_values.sort()
    #x = np.arange(1,len(param_values)+1)
    return grid,grid.cv_results_['params'],grid.cv_results_['mean_test_score']
    
grid_search(X_train,y_train,model,param_grid,cv=3)

