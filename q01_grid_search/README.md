# Gridsearch for Randomforest

* Let us check for every params and what is the mean and std.
* We have made your life simpler we have given the parameter to be tunned for Randomforest:

## Write a Function `grid_search` that:
- Fit the `GridSearchCV` model on X_train and y_train
- Will take five parameters and return the grid_score for each params 
***
param_grid = 

              {"max_features": ['sqrt',4,"log2"],
             "n_estimators" : [10, 50, 120],
             "max_depth" : [None, 40, 20, 10],
             "max_leaf_nodes": [5, 10, 2]
             }
***

### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- | 
| X_train | pandas dataframe | compulsory | | X_train dataframe |
| y_train | pandas dataframe | compulsory | | y_train dataframe |
| model |  | compulsory | | model to be fitted |
| param_grid | | compulsory | | parameter to be tunned |
| CV | | optional | 3 | Fold to be given |



### Returns:
| Return | dtype | description |
| --- | --- | --- | 
| variable1 | list | returns list containing values of param_grid at each iteration|
| variable2 | numpy.ndarray | returns score for each param_grid |



Note :-
- Specify the model(RandomForestClassifier) to be used before defining the function and set `oob_score` as True 
and `random_state` as 9.