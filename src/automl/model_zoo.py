# The goal of this file is to select candidate models and return initialized model objects.

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

class ModelZoo:

    def get_models(self, task):

        if task == "classification":

            return {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }

        elif task == "regression":

            return {
                "Linear Regression": LinearRegression(), 
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }

        else:

            raise ValueError("Unsupported task type")

