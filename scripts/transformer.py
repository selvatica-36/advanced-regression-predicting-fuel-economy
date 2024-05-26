from scipy import stats
from typing import List
import numpy as np
import pandas as pd


class DataTransform:
    """
    A class for performing various transformations on a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to transform.

    Example:
    ```
    transformer = DataTransform(my_dataframe)
    ```

    """    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataTransform object with a DataFrame. Used internally when an instance of the call is called.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to transform.

        """
        self.df = dataframe.copy()

    def convert_to_type(self, column_name: str, data_type: str, ignore_errors: bool = True) -> pd.DataFrame:
        """
        Convert a specific column in the DataFrame to the specified data type.

        Parameters:
        - column_name (str): Name of the column to be converted.
        - data_type (str): Target data type.
        - ignore_errors (bool, optional): Whether to ignore errors during conversion. Default is True.

        """
        data_type = data_type.lower()
        if ignore_errors == True:
            error_statement = ["coerce", "ignore"]
        elif ignore_errors == False:
            error_statement = ["raise", "raise"]
        else:
            print("Error: the parameter 'ignore_errors' is a bool and can only be True or False.")
        # Convert column to datatype:
        try:
            if data_type in ["datetime", "date"]:
                self.df[column_name] = pd.to_datetime(self.df[column_name], errors=error_statement[0])
            elif data_type in ["str", "int", "float", "bool", "int64", "float64"]:
                data_type = data_type.replace("64", "")
                self.df[column_name] = self.df[column_name].astype(data_type, errors=error_statement[1])
            elif data_type == "categorical":
                self.df[column_name] = pd.Categorical(self.df[column_name])
            else:
                print(f"Error: data type {data_type} not supported. Check docstrings or call help for more information.")
        except Exception as e:
            print(f"Error converting column '{column_name}' to type '{data_type}': {e}")
        # TODO You could move a lot of what is in the this method to a dictionary mapping of the function and datatypes
        # it will keep your code cleaner but I really like the idea. 
        # you could reduce the size of the this method but place the conversion part of the code in another method and calling it here.
        # There are other ways to do this with dictionaries as well but it should reduce the overall size of your method doing it this way

    def convert_month_to_period(self, column_name: str) -> pd.DataFrame:
        """
        Convert a column representing months to a period format.

        Parameters:
        - column_name (str): Name of the column to be converted.

        Returns:
        - pd.DataFrame: A copy of the DataFrame with the converted column.

        """
        try:
            self.df[column_name] = self.df[column_name].astype(str)
            self.df['month'] = self.df['month'].str.lower()
            # NOTE I would actually move your mappings into a separate file here and import it just to keep it cleaner
            month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'june': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            self.df[column_name] = self.df[column_name].map(month_map)
            self.df[column_name] = pd.to_datetime(self.df[column_name], format='%m', errors='coerce').dt.to_period('M')
        except Exception as e:
            print(f"Error converting 'month' column to period: {e}")
        return self.df.copy()
    
    def convert_month_to_datetime(self, column_name: str) -> pd.DataFrame:
        """
        Convert a column representing months to a datetime format.

        Parameters:
        - column_name (str): Name of the column to be converted.

        Returns:
        - pd.DataFrame: A copy of the DataFrame with the converted column.

        """
        try:
            self.df[column_name] = self.df[column_name].astype(str)
            self.df['month'] = self.df['month'].str.lower()
            # NOTE I would actually move your mappings into a separate file here and import it just to keep it cleaner
            month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'june': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            self.df[column_name] = self.df[column_name].map(month_map)
            self.df[column_name] = pd.to_datetime(self.df[column_name], format='%m', errors='coerce')
        except Exception as e:
            print(f"Error converting 'month' column to datetime: {e}")
        return self.df.copy()
    
    def convert_columns(self, column_list: List[str], data_type: str, ignore_errors: bool = True) -> pd.DataFrame:
        """
        Convert multiple columns to the specified data type.

        Parameters:
        - column_list (List[str]): List of column names to be converted.
        - data_type (str): Target data type.
        - ignore_errors (bool, optional): Whether to ignore errors during conversion. Default is True.

        Returns:
        - pd.DataFrame: A copy of the DataFrame with the converted columns.

        """
        for column in column_list:
            self.convert_to_type(column, data_type, ignore_errors)
        return self.df.copy()
    
    def rename_column(self, col_name: str, new_col_name: str) -> pd.DataFrame:
        """
        Rename a column in the DataFrame.

        Parameters:
        - col_name (str): Current name of the column.
        - new_col_name (str): New name for the column.

        Returns:
        - pd.DataFrame: A copy of the DataFrame with the renamed column.

        """
        self.df.rename(columns={col_name: new_col_name}, inplace=True)
        return self.df.copy()
    
    def round_float(self, column: str, decimal_places: int) -> pd.DataFrame:
        """
        Round the values in a column to a specified number of decimal places.

        Parameters:
        - column (str): Name of the column to be rounded.
        - decimal_places (int): Number of decimal places to round to.

        Returns:
        - pd.DataFrame: A copy of the DataFrame with the rounded column.

        """
        self.df[column] = self.df[column].apply(lambda x: round(x, decimal_places))
        return self.df.copy()
    
    def impute_nulls(self, column_list: List[str], method: str) -> pd.DataFrame:
        """
        Impute null values in specified columns using a specified method.

        Parameters:
        - column_list (List[str]): List of column names to impute null values.
        - method (str): Imputation method ('mean', 'median', or 'mode').

        Returns:
        - pd.DataFrame: A copy of the DataFrame with null values imputed.

        Example:
        ```
        df = transformer.impute_nulls(['column1', 'column2'], 'mean')
        ```

        """
        method = method.lower()
        valid_methods = ['mean', 'median', 'mode']

        try:
            if method not in valid_methods:
                raise ValueError(f"Invalid imputation method. Method can only be one of: {', '.join(valid_methods)}")

            for column in column_list:
                if method == 'median':
                    self.df[column] = self.df[column].fillna(self.df[column].median())
                elif method == 'mean':
                    self.df[column] = self.df[column].fillna(self.df[column].mean())
                elif method == 'mode':
                    self.df[column] = self.df[column].fillna(self.df[column].mode()[0])

        except ValueError as ve:
            print(f"Error: {ve}. Please check that you have provided a list of column names formatted as strings.")
        return self.df.copy()

    def impute_nulls_with_median(self, column_list: List[str]) -> pd.DataFrame:
        """
        Impute null values in specified columns using the median.

        Parameters:
        - column_list (List[str]): List of column names to impute null values.

        Returns:
        - pd.DataFrame: A copy of the DataFrame with null values imputed.

        Example:
        ```
        df = transformer.impute_nulls_with_median(['column1', 'column2'])
        ```

        """
        for column in column_list:
            self.df[column] = self.df[column].fillna(self.df[column].median())
        return self.df
    
    def impute_nulls_with_mean(self, column_list: List[str]) -> pd.DataFrame:
        """
        Impute null values in specified columns using the mean.

        Parameters:
        - column_list (List[str]): List of column names to impute null values.

        Returns:
        - pd.DataFrame: A copy of the DataFrame with null values imputed.

        Example:
        ```
        df = transformer.impute_nulls_with_mean(['column1', 'column2'])
        ```

        """
        for column in column_list:
            self.df[column] = self.df[column].fillna(self.df[column].mean())
        return self.df
    
    def impute_nulls_with_mode(self, column_list: List[str]) -> pd.DataFrame:
        """
        Impute null values in specified columns using the mode.

        Parameters:
        - column_list (List[str]): List of column names to impute null values.

        Returns:
        - pd.DataFrame: A copy of the DataFrame with null values imputed.

        Example:
        ```
        df = transformer.impute_nulls_with_mode(['column1', 'column2'])
        ```

        """
        for column in column_list:
            self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
        return self.df
    
    def impute_nulls_with_zeros(self, column_list):
        """
        Impute null values in specified columns with zeros.

        Parameters:
        - column_list (List[str]): List of column names to impute null values.

        Returns:
        - pd.DataFrame: A copy of the DataFrame with null values imputed.

        Example:
        ```
        df = transformer.impute_nulls_with_zeros(['column1', 'column2'])
        ```

        """
        for col in column_list:
            self.df[col] = self.df[col].fillna(0)

    def log_transform(self, column_list: List[str]) -> pd.DataFrame:
        """
        Apply a log transformation to specified columns.

        Parameters:
        - column_list (List[str]): List of column names to transform.

        Returns:
        - pd.DataFrame: Transformed DataFrame.

        Example:
        ```
        transformed_df = transformer.log_transform(['column1', 'column2'])
        ```

        """
        for col in column_list:
            self.df[col] = self.df[col].map(lambda i: np.log(i) if i > 0 else 0)
        return self.df
    
    def boxcox_transform(self, column_list: List[str]) -> pd.DataFrame:
        """
        Apply a Box-Cox transformation to specified columns.

        Parameters:
        - column_list (List[str]): List of column names to transform.

        Returns:
        - pd.DataFrame: Transformed DataFrame.

        Example:
        ```
        transformed_df = transformer.boxcox_transform(['column1', 'column2'])
        ```

        """
        for col in column_list:
            boxcox_population, lambda_values = stats.boxcox(self.df[col])
            self.df[col] = boxcox_population
        return self.df
    
    def yeo_johnson_transform(self, column_list: List[str]) -> pd.DataFrame:
        """
        Apply a Yeo-Johnson transformation to specified columns.

        Parameters:
        - column_list (List[str]): List of column names to transform.

        Returns:
        - pd.DataFrame: Transformed DataFrame.

        Example:
        ```
        transformed_df = transformer.yeo_johnson_transform(['column1', 'column2'])
        ```

        """
        for col in column_list:
            nonzero_values = self.df[col][self.df[col] != 0]
            yeojohnson_values, lambda_value = stats.yeojohnson(nonzero_values)
            self.df[col] = self.df[col].apply(lambda x: stats.yeojohnson([x], lmbda=lambda_value)[0] if x != 0 else 0)
        return self.df
    
