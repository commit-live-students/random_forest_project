# %load q01_grid_search/build.py
# Default imports

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
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
model=RandomForestClassifier(random_state=9,oob_score=True)
def grid_search(X_train,y_train,model,param_grid,cv=3):
    gs=GridSearchCV(model,param_grid,cv=cv,n_jobs=-1)
    gs.fit(X_train,y_train)
# Write your solution here :
    grid_dict=gs.grid_scores_
    list_mean=[]
    list_mean2=[]
    for i in range(len(grid_dict)):
         list_mean.append(grid_dict[i][1])
         list_mean2.append(grid_dict[i][0])


    grid_score=np.array(list_mean)
    grid_param=tuple(list_mean2)
    return gs,grid_param,grid_score
