# from automl.data_loader import DataLoader
# from automl.preprocessor import Preprocessor

# loader = DataLoader(path=r"data\Salary_Regression.csv", target="Salary")

# X, y, task = loader.load()

# processor = Preprocessor()

# pipeline = processor.build(X)

# X_processed = pipeline.fit_transform(X)

# print(X_processed.shape)

import pandas as pd
from automl.validator import Validator

df = pd.DataFrame({
     "age": [20, 30, 40],
    "salary": [3000, 5000, 7000],
    "target": [0, 1, 1]
})

validator = Validator(df, target="worng")

validator.validate()

