# This will be like the brain of the AutoML

from .data_loader import DataLoader
from .validator import Validator
from .preprocessor import Preprocessor
from .feature_engineer import FeatureEngineer
from .model_zoo import ModelZoo
from .evaluator import Evaluator


from sklearn.model_selection import train_test_split

class AutoML:

    def __init__(self, data_path, target):
        # the data path the user will put and target column

        self.data_path = data_path
        self.target = target



        self.data_loader = DataLoader(path = data_path, target = target)

        self.validator = None

        self.feature_engineer = FeatureEngineer()

        self.preprocessor = Preprocessor()

        self.model_zoo = ModelZoo()

        self.evaluator = None


    def fit(self):

        # 1. Load The Data

        df, task = self.data_loader.load()
        print("Detected task:", task)

        # 2. Validate The Data

        self.validator = Validator(df, self.data_loader.target)

        self.validator.validate()


        # 3. Split X and y

        X, y = self.data_loader.split_data(df)

        # 4. Feature Engineering 

        self.feature_engineer.fit(X)

        X = self.feature_engineer.transform(X)

        # 5. Preprocessing 

        pipeline = self.preprocessor.build(X)

        X = pipeline.fit_transform(X)


        # 6. Train / Test Split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # 7. Get The Models 

        models = self.model_zoo.get_models(task)

        # 8. Evaluate The Retrieved Models

        self.evaluator = Evaluator(task)

        results = self.evaluator.evaluate(models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


        # 9. Return The Results 

        return {
            "best_model_name": self.evaluator.best_model,
            "best_model": self.evaluator.best_model,
            "best_score": self.evaluator.best_score,
            "all_results": results
        }

