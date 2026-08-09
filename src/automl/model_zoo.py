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
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42)
            }

        else:

            raise ValueError(f"Unknown task: {task}")


    def get_search_space(self, model_name, trial):

        if model_name == "Random Forest":

            return {
                    "n_estimators": trial.suggest_int(
                    "n_estimators",
                    100,
                    500
                ),

                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    20
                ),

                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    2,
                    10
                ),

                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    5
                )
            }


        elif model_name == "Gradient Boosting":

            return {
                    "n_estimators": trial.suggest_int(
                    "n_estimators",
                    50,
                    300
                ),

                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.3
                ),

                "max_depth": trial.suggest_int(
                    "max_depth",
                    2,
                    10
                ),

                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    2,
                    10
                ),

                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    5
                )
            }


        elif model_name == "Linear Regression":

            return {}

        else:

            raise ValueError(f"Unknown model: {model_name}")
    

