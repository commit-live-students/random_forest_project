# %load q01_grid_search/build.py
# Default imports

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer

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
def grid_search(X_train,y_train,model, param_grid,cv=3):      
    clf = model
    np.random.seed(9)
    parameters = param_grid
    acc_scorer = make_scorer(accuracy_score)
    
    # Run the grid search
    grid_obj = GridSearchCV(clf, param_grid, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, y_train)
    variable1=list()
    variable3=list()
    for i in grid_obj.grid_scores_:
        variable1.append(i[0])
        variable3.append(i[1])
    variable2=np.array(variable3)
    return grid_obj,variable1,variable2


