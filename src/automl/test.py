from automl.data_loader import DataLoader
from automl.preprocessor import Preprocessor


X, y, task = DataLoader(path="./Salary_dataset.csv", target="Salary")

processor = Preprocessor()

pipeline = processor.build(X)

X_processed = pipeline.fit_transform(X)

print(X_processed.shape)

