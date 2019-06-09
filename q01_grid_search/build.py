# %load q01_grid_search/build.py
# Default imports

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,accuracy_score

loan_data = pd.read_csv('data/loan_prediction.csv')
X_bal = loan_data.iloc[:, :-1]
y_bal = loan_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.33, random_state=9)
param_grid = {'max_features': ['sqrt', 4, 'log2'],
              'n_estimators': [10, 50, 120],
              'max_depth': [40, 20, 10],
              'max_leaf_nodes': [5, 10, 2]}

model = RandomForestClassifier(oob_score=True, random_state=9)


# Write your solution here :
def grid_search(X_train, y_train, model, param_grid, cv=3):
    clf = GridSearchCV(model,param_grid,cv=cv)
    clf.fit(X_train,y_train)

    
    results = pd.DataFrame(clf.cv_results_)
    model1 = clf
    variable1 = list(results['params'])
    variable2 = np.array(results['mean_test_score'])
    return model1,variable1,variable2
    


#print(grid_search(X_train,y_train,model,param_grid,cv = 3))

