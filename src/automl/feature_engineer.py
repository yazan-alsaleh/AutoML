# This fill will be Given a clean dataset, can I create better features automatically?


# the class follows exactly the same design as Scikit-Learn estimators.
# Scikit-Learn estimators are any Python object that learns from data to fit a machine learning model, likek linear regression, randome forest...
# Every estimator implements a unified API centered around the .fit(X, y) method.

import pandas as pd

class FeatureEngineer:

    def __init__(self):
        # These variables store what the model learns. Think of them as memory.
        self.constant_columns = [] # if a columns values dos not change, the model will learn this and store its title in this array. 
        # Later during the transform step it knows to remove this column because its values are not chaning and will not improve the model. 
        self.duplicate_columns = [] # Store the columns that contain exactly the same values so remove one of them.


    def fit(self, X): # looks at the data and learns information from it.
        pass


    def transform(self, X): # uses the information learned by fit() to modify the data.
        pass


    def fit_transform(self, X): # learn + apply. See notebook_1.ipynm in /notebooks
        pass


