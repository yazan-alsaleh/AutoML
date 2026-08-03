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

        # Get the inputs (x)
        
        X = df.drop(columns = [self.target]) # Get all the columns except the trager 

        y = df[self.target] # Get the target / to be predicted column from the data frame


        # Instead of asking the user: is this problem classification or regression 

        # Detect what the task should be 
        task = self.detect_task(y)

        return X, y, task


