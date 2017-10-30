# Default imports

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
loan_data = pd.read_csv('data/loan_prediction.csv')
X_bal = loan_data.iloc[:, :-1]
y_bal = loan_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.33, random_state=9)
param_grid = {"max_features": ['sqrt', 4, "log2"],
              "n_estimators": [10, 50, 120],
              "max_depth": [40, 20, 10],
              "max_leaf_nodes": [5, 10, 2]}

def grid_search(X_train,y_train,model,param_grid,cv=3):

    amodel = GridSearchCV(model,param_grid,cv=cv)
    amodel.fit(X_train,y_train)
    param_list=[]
    accuracy_list=[]
    for i in range(len(amodel.grid_scores_)):
        a = amodel.grid_scores_[i]
        param_list.append(a[0])
        accuracy_list.append(a[1])

    return amodel,tuple(param_list),np.array(accuracy_list)


# Write your solution here :
