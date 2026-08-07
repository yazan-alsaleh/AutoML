

from automl.automl import AutoML

automl = AutoML(data_path="data\Loan_Classification.csv", target="loan_status")


result = automl.fit()

print(result.best_model_name)
print(result.best_score)
print(result.all_results)
print(result.best_model)

