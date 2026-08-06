# the part that actually makes your system "AutoML". it test all models and return the best one

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


class Evaluator:

    def __init__(self, task):

        self.task = task

        self.results = {}

        self.best_model = None
        self.best_score = None


    def evaluate(self, models, X_train, X_test, y_train, y_test):

        for name, model in models.items(): # model comes from model zoo
                                # item will give both model name and model object

            print(f"Training {name}...")


            # Train the model
            model.fit(X_train, y_train)


            # Predict
            predictions = model.predict(X_test)

            # Get the score of the model on testing data

            if self.task == "classification":

                score = accuracy_score(y_test, predictions)

            else: # regression

                score = r2_score(y_test, predictions)


            # Save the score of each model to show them later in the final report 
            self.results[name] = score


        self._find_best_model()

        return self.results




    def _find_best_model(self):

        self.best_model = max(self.results, key = self.results.get)
        # max by default compares keys (the models names), we want to compare values, so use (key) argument, to compare the values not the keys of the dictionary.
        # max here is only to compare, it will store the key (the model name).

        self.best_score = self.results[self.best_model]
        # Go inside the dictionary and find the key of the best model and store its score.

