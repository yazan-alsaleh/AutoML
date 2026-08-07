# without this file the system (the Evaluator) will use the default values for the parameters when using the models. 
# Instead AutoML should automatically search for better values (hyperparameter combination).


# See notebook_2.ipynb to understand automatic hyperparameter combination.

# What the tuner should do: 

# Example:
# Input: RandomForestRegressor
# Output: RandomForestRegressor(n_estimators = 350, max_depth = 18, min_samples_split = 4)

import optuna

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
class Tuner:

    def __init__(self, task):

        self.task = task

        self.best_params = None # best hyperparameter combination
        self.best_model = None # we will store here the final tuned model

    def objective(self, trail, model_name, X, y):

        # trail represents one experiment Optuna might create 


        if model_name == "Random Forest": 

            n_estimators = trail.suggest_int("n_estimators", 100, 500)
            # suggest_int() means Optuna, choose an integer between 100 and 500 for "n_estimators" and same for others
            max_depth = trail.suggest_int("max_depth", 3, 20)

            min_samples_split = trail.suggest_int("min_samples_split", 2, 10)

            min_samples_leaf = trail.suggest_int("min_samples_leaf", 1, 5)

            model = RandomForestRegressor(n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          random_state=42)

        elif model_name == "Gradient Boosting":

            n_estimators = trail.suggest_int("n_estimators", 50, 300)

            learning_rate = trail.suggest_float("learning_rate", 0.01, 0.3)

            max_depth = trail.suggest_int("max_depth", 2, 10)

            min_samples_split = trail.suggest_int("min_samples_split", 2, 10)

            min_samples_leaf = trail.suggest_int("min_samples_leaf", 1, 5)

            model = GradientBoostingRegressor(n_estimators=n_estimators, 
                                              learning_rate=learning_rate,
                                              max_depth=max_depth,
                                              min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              random_state=42)








