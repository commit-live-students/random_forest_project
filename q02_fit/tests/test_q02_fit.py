from unittest import TestCase
from ..build import fit
from sklearn.ensemble import RandomForestClassifier
from greyatomlib.random_forest_project.q01_grid_search.build import grid_search
from sklearn.model_selection import train_test_split
from inspect import getargspec
import pandas as pd
import numpy

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

class TestFit(TestCase):
    def test_fit(self):  # Input parameters tests
        args = getargspec(fit)
        self.assertEqual(len(args[0]), 2, "Expected argument(s) %d, Given %d" % (2, len(args[0])))

    def test_fit_default(self):  # Input parameter defaults
        args = getargspec(fit)
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

    def test_fit_conf_matrix_type(self):  # Return data types
        conf_matrix, class_report, accuracy = fit(X_test, y_test)
        self.assertIsInstance(conf_matrix, numpy.ndarray,
                              "Expected data type for return value is `numpy.ndarray`, you are returning %s" % (
                                  type(conf_matrix)))

    def test_fit_accuracy_type(self):
        conf_matrix, class_report, accuracy = fit(X_test, y_test)
        self.assertIsInstance(accuracy, numpy.float64,
                              "Expected data type for return value is `numpy.float64`, you are returning %s" % (
                                  type(accuracy)))

    def test_fit_accuracy_values(self): # Return value tests
        conf_matrix, class_report, accuracy = fit(X_test, y_test)
        self.assertAlmostEqual(accuracy, 0.778325123153, 3, "Return value of accuracy does not match expected "
                                                                  "value")

