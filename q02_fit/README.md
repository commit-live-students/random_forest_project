#  Write a function to fit a model 

* We know now how does param grid work, now lets build a model that will predict on test set.

## Write a Function `fit` that:
* Uses Grid from the above function and perdict on test set.
* Now check the predicted output with y_test, the accuracy_score, classification_report, confusion_matrix


### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- | 
| X_test | pandas dataframe | compulsory | | X_test dataframe |
| y_test | pandas dataframe | compulsory | | y_test dataframe |

### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| variable1 | numpy.float64 | confusion_matrix of the model |
| variable2 | unicode | classification_report of the model |
| variable3 |numpy.ndarray | accuracy_score of the model |

