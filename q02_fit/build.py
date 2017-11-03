# Default imports

import pandas as pd
from greyatomlib.random_forest_project.q01_grid_search.build import grid_search
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


loan_data = pd.read_csv('data/loan_prediction.csv')
X_bal = loan_data.iloc[:, :-1]
y_bal = loan_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.33, random_state=9)
rfc = RandomForestClassifier(oob_score=True, random_state=9)
param_grid = {"max_features": ['sqrt', 4, "log2"],
              "n_estimators": [10, 50, 120],
              "max_depth": [40, 20, 10],
              "max_leaf_nodes": [5, 10, 2]}

grid, grid_param, grid_score = grid_search(X_train, y_train, rfc, param_grid, cv=3)


# Write your solution here :
def fit(X_test, y_test):
    y_pred = grid.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    con_matrix = confusion_matrix(y_test, y_pred)
    return con_matrix,class_report,acc_score
