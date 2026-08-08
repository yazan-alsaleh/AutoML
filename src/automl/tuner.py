# without this file the system (the Evaluator) will use the default values for the parameters when using the models. 
# Instead AutoML should automatically search for better values (hyperparameter combination).


# See notebook_2.ipynb to understand automatic hyperparameter combination.

# What the tuner should do: 

# Example:
# Input: RandomForestRegressor
# Output: RandomForestRegressor(n_estimators = 350, max_depth = 18, min_samples_split = 4)

import optuna

# cross_val_score is used to evaluate each hyperparameter combination using cross-validation method 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
class Tuner:

    def __init__(self, task):

        self.task = task

        self.best_params = None # best hyperparameter combination
        self.best_model = None # we will store here the final tuned model

    # the objective method means Optuna give me a set of hyperparameters, and I'll tell you how good they are.
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

        elif model_name == "Linear Regression":

            model = LinearRegression()

        else:

            raise ValueError(f"Unknown model: {model_name}")

        score = cross_val_score(model, X, y, cv=5, scoring="r2")

        return score.mean()



    def tune(self, model_name, X, y):
        # model_name --> which model you want to tune
        # X --> input
        # y --> target

        # example: Find the best hyperparameters for Random Forest using X and y


        # A study is an Optuna object that contains / manages the whole tuning process.
        study = optuna.create_study(direction="maximize") # maximize means Find the trial with the highest score

        def objective(trail):
            return self.objective(trail, model_name, X, y)
        
        # Start the optimization It tells Optuna: Run my objective function 20 times and find the best result
        study.optimize(objective, n_trials=20)

        self.best_params = study.best_params

        self.best_model = self.build_model(model_name, self.best_params)

        return self.best_params


    def build_model(self, model_name, params):
        # model_name is which model we want to create 
        # params contains the hyperparameters from Optuna

        if model_name == "Random Forest":

            return RandomForestRegressor(**params, random_state=42)
            # Because Optuna returnsa a dictionary that contains the values of each parameter for the model,
            # we use ** to convert it to actual parameter / arguments and its value, like (n_estimators = 300)

        elif model_name == "Gradient Boosting":

            return GradientBoostingRegressor(**params, random_state=42)

        elif model_name == "Linear Regression":

            return LinearRegression()
            # Because your current LinearRegression doesn't need the hyperparameters


        


