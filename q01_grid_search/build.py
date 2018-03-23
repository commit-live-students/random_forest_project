# Default imports

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
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
clf = RandomForestClassifier(oob_score=True, random_state=9)

# Write your solution here :
def grid_search(X_train, y_train, clf, param_grid, CV):
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=CV)
    #optimized_grid = model(clf, cv_params = param_grid, cv = CV)
    grid_search.fit(X_train, y_train)
    return grid_search, (grid_search.cv_results_['params']), (grid_search.cv_results_['mean_test_score'])

grid_search(X_train, y_train, clf, param_grid=param_grid, CV=3)
