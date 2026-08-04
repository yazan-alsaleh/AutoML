from automl.data_loader import DataLoader
from automl.preprocessor import Preprocessor

loader = DataLoader(path=r"src\automl\Salary_dataset.csv", target="Salary")

X, y, task = loader.load()

processor = Preprocessor()

pipeline = processor.build(X)

X_processed = pipeline.fit_transform(X)

print(X_processed.shape)

