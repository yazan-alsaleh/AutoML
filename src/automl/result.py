# This class is simply a container for the results from evaluator
# so later if you want to add more information you can add them as new attributes

class AutoMLResult:

    def __init__(self, best_model_name, best_model, best_score, all_results):


        self.best_model_name = best_model_name
        self.best_model = best_model
        self.best_score = best_score
        self.all_results = all_results


