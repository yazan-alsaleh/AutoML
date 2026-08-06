# Here we will implement the dataLoader

import pandas as pd


# Using a class instead of standalone functions lets the loader store configuration
# (like the dataset path and target column) and makes it easy to extend later.
class DataLoader:
    
    def __init__(self, path, target):

        self.path = path # dataset path
        self.target = target # target column that we want to predict 


    # The main method of the class

    def load(self):


        df = pd.read_csv(self.path) # read the CSV


        # Instead of asking the user: is this problem classification or regression 
        # Detect what the task should be 
        task = self.detect_task(df[self.target])

        return df, task


    def split_data(self, df):

        # Get the inputs (x)
                
        X = df.drop(columns = [self.target]) # Get all the columns except the trager. Becauses pandas receive a list we add [ ]
        y = df[self.target] # Get the target / to be predicted column from the data frame
        

        return X, y

    # Get the problem type
    def detect_task(self, y):

        if y.dtype == "object": # like the rows in the column is: dog, dog, cat, dog. These are objects / labels
            return "classification"
        
        if y.nunique() < 20: 
            return "classification"
        # nunique() counts how many different values exist in the target column.
        # example: dog, dog, cat, dog --> it will return 2.

        # Why checking if it is less than 20? Because mostly there could be 5, 10, at max 20 unique value
        
        else:
            return "regression"


# Testing the calss 


# loader = DataLoader(path="loan.csv", target="loan_status")
# X, y, task = loader.load()
# print(task)