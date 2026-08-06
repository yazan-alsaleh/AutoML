from automl.automl import AutoML

automl = AutoML(data_path="data\Salary_Regression.csv", target="Salary")


result = automl.fit()

print(result)

