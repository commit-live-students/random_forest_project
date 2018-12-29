# %load q01_grid_search/build.py
# Default imports

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
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

rfc = RandomForestClassifier(oob_score=True, random_state=9)
# Write your solution here :
def grid_search(X_train,y_train,model,param_grid,cv=3):
    grid_search_1 = GridSearchCV(estimator=model,param_grid=param_grid,cv=cv)
    best_rf = grid_search_1.fit(X_train,y_train)
    return best_rf,best_rf.cv_results_['params'],best_rf.cv_results_['mean_test_score']

grid_search(X_train,y_train,rfc,param_grid,3)


