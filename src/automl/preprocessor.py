# The Preprocessor prepares your raw data so that machine learning models can understand it.

# A machine learning model cannot directly work with messy real-world data. It needs:
# numbers instead of text
# missing values handled
# features scaled properly

# The preprocessor automates this.



# ColumnTransformer allows you to apply different transformations to different columns / preprocess different columns differently.
from sklearn.compose import ColumnTransformer # its like traffic controller
# How do I connect preprocessing and model training together
from sklearn.pipeline import Pipeline
# It fills missing values with a reasonable value.
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler # For scaling the numbers
from sklearn.preprocessing import OneHotEncoder # Convert categories into numbers


class Preprocessor:

    def build(self, X):

        numerical = X.selext_dtypes(include = ["int64", "float64"]).columns # returns columns with numeric types.

        categorical = X.select_dtypes(include = [object]).columns # returns columns with categorical types.


        # Numerical pipeline
        num_pipeline = Pipeline([
            (   # first: fill the missing values in that column
                "imputer", SimpleImputer(strategy="median")
            ),
            (   # second: scale the numbers
                "scalar", StandardScaler()
            )
        ])


        # Categorical pipeline
        cat_pipeline = Pipeline([
            (   # First: Replace NaN values with the most common
                "imputer", SimpleImputer(strategy="most_frequent") 
            ),
            (
                # Convert the text into numbers
                "encoder", OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformer = ColumnTransformer([
            (   # "num" here is the name of the pipeline that will represent the numerical values
                "num", num_pipeline, numerical # Use the pipeline called "num" on numerical columns that we have selected at the beginning of the method 
            ),
            (   # "cat" here is the name of the pipeline that will represent the numerical values
                "cat", cat_pipeline, categorical # Use the pipeline called "cat" on categorical columns that we have selected at the beginning of the method 
            )
        ])

        return transformer


