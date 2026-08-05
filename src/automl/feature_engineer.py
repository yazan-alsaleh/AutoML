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

        for column in X.columns: # for each column in the input data (features)

            if X[column].nunique() <= 1: # count and if less than 1 or one means its unique
                self.constant_columns.append(column) # save the column name


        duplicates = X.T.duplicated() # take the transpose of the X inputs / features because duplicated() checks rows by default, but you want to find duplicate columns.

        self.duplicate_columns = []

        for column, is_duplicate in zip(X.columns, duplicates):
            if is_duplicate:
                self.duplicate_columns.append(column)

        # X.columns will contain --> ["age", "age_copy", "salary"]
        # duplicates: will contain --> [False, True, False]

        # when using in zip() --> ("age", False)("age_copy", True)("salary", False), on the left the column name and on the right if the column duplicated or not. 

        # now when the loop starts, column will be: column = "age" and is_duplicate = False --> Do nothing because its not duplicated.
        # second loop column will be: column = "age_copy" and is_duplicate = True --> keep it because its duplicated.


        return self # return the feature engineering object

    

    def transform(self, X): # uses the information learned by fit() to modify the data.
        pass


    def fit_transform(self, X): # learn + apply. See notebook_1.ipynm in /notebooks
        pass


