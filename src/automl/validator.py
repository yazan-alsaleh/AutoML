# this file will check if the data is usable before we spend time preprocessing or training models.


class Validator:

    def __init__(self, df, target):

        self.df = df
        self.target = target


    # This will be the main method that will have all other methods inside it for validation
    # When the user call validator.validate all validation methods will be called automatically 
    def validate(self):

        self.check_empty()
        self.check_target_exists()
        self.check_missing_target()
        self.check_columns()
        self.check_duplicate_columns()

        print("Validation passed!")


    def check_empty(self):

        if self.df.empty:
            raise ValueError("Dataset is empty.")



    def check_target_exists(self):

        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' does not exist in the given dataset.")



    # if there any missing values in the target column
    def check_missing_target(self):

        missing = self.df[self.target].isna().sum()

        if missing > 0:
            raise ValueError("Target column contains missing values.")



    def check_columns(self):

        if len(self.df.columns) < 2:
            raise ValueError("Dataset needs at least one feature and one target column.")


    def check_duplicate_columns(self):

        # get the duplicates
        duplicates = self.df.columns[
            self.df.columns.duplicated()
        ]

        if len(duplicates) > 0:
            raise ValueError(f"Duplicate columns found: {list(duplicates)}")

