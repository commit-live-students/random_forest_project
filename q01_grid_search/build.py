# %load q01_grid_search/build.py
# Default imports

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

loan_data = pd.read_csv('data/loan_prediction.csv')
X_bal = loan_data.iloc[:, :-1]
y_bal = loan_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.33, random_state=9)
param_grid = {"max_features": ['sqrt', 4, "log2"],
              "n_estimators": [10, 50, 120],
              "max_depth": [40, 20, 10],
              "max_leaf_nodes": [5, 10, 2]}

model = RandomForestClassifier(oob_score=True, random_state=9)
# Write your solution here :
def grid_search(X_train,y_train,model,param_grid,cv=3):


    #acc_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(model, param_grid,cv=cv)
    grid_obj.fit(X_train,y_train)

    #print(grid_obj.get_params)
    #print(np.shape(grid_obj.cv_results_['params']))
    #print(grid_obj.cv_results_['mean_test_score'])
    #print(grid_obj.cv_results_['mean_test_score'].shape)
    #mean_train_scores = grid_obj.cv_results_['mean_train_score']
    #print(grid_obj.grid_scores_)
    return grid_obj,grid_obj.cv_results_['params'],grid_obj.cv_results_['mean_test_score']

grid, grid_param, grid_score=grid_search(X_train,y_train,model,param_grid,cv=3)
#print(grid_score)
expected_param = [{'max_features': 'sqrt', 'max_leaf_nodes': 5, 'n_estimators': 10, 'max_depth': 40},
                  {'max_features': 'sqrt', 'max_leaf_nodes': 5, 'n_estimators': 50, 'max_depth': 40},
                          {'max_features': 'sqrt', 'max_leaf_nodes': 5, 'n_estimators': 120, 'max_depth': 40},
                          {'max_features': 'sqrt', 'max_leaf_nodes': 10, 'n_estimators': 10, 'max_depth': 40},
                          {'max_features': 'sqrt', 'max_leaf_nodes': 10, 'n_estimators': 50, 'max_depth': 40},
                          {'max_features': 'sqrt', 'max_leaf_nodes': 10, 'n_estimators': 120, 'max_depth': 40},
                          {'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 10, 'max_depth': 40},
                          {'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 50, 'max_depth': 40},
                          {'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 120, 'max_depth': 40},
                          {'max_features': 4, 'max_leaf_nodes': 5, 'n_estimators': 10, 'max_depth': 40},
                          {'max_features': 4, 'max_leaf_nodes': 5, 'n_estimators': 50, 'max_depth': 40},
                          {'max_features': 4, 'max_leaf_nodes': 5, 'n_estimators': 120, 'max_depth': 40},
                          {'max_features': 4, 'max_leaf_nodes': 10, 'n_estimators': 10, 'max_depth': 40},
                          {'max_features': 4, 'max_leaf_nodes': 10, 'n_estimators': 50, 'max_depth': 40},
                          {'max_features': 4, 'max_leaf_nodes': 10, 'n_estimators': 120, 'max_depth': 40},
                          {'max_features': 4, 'max_leaf_nodes': 2, 'n_estimators': 10, 'max_depth': 40},
                          {'max_features': 4, 'max_leaf_nodes': 2, 'n_estimators': 50, 'max_depth': 40},
                          {'max_features': 4, 'max_leaf_nodes': 2, 'n_estimators': 120, 'max_depth': 40},
                          {'max_features': 'log2', 'max_leaf_nodes': 5, 'n_estimators': 10, 'max_depth': 40},
                          {'max_features': 'log2', 'max_leaf_nodes': 5, 'n_estimators': 50, 'max_depth': 40}]

expected_score = [0.74939173, 0.74209246, 0.756691, 0.75425791, 0.75425791, 0.75912409,
                  0.71289538, 0.71046229, 0.71046229, 0.75912409, 0.76155718, 0.76155718,
                          0.74209246, 0.75912409, 0.756691, 0.76642336, 0.76642336, 0.76642336,
                          0.74939173, 0.74209246]


for i in range(len(grid_param)):
            if grid_param[i] in expected_param:
                expected_index = expected_param.index(grid_param[i])

                print(i,grid_score[i],expected_score[expected_index])
