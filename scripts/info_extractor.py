from tabulate import tabulate
from typing import Optional, List
import numpy as np
import pandas as pd


class DataFrameInfo:
    """
    Initialize the DataFrameInfo object with a DataFrame. Used internally when an instance of the call is called.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to analyze.

    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataFrameInfo object with a DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to analyze.

        """
        self.df = dataframe.copy()

    def get_slice(self, columns=None) -> pd.DataFrame:
        """
        Get a subset of the DataFrame based on specified columns.

        Parameters:
        - columns: Columns to include in the subset. Can be provided as 
                   a list of column names formatted as string,
                   as a string if only one column is chosen, 
                   or as a tuple or two integers indicating the positions of the columns in the dataframe.
                   If a tuple is provided, the counting starts from zero. 
                   The second integer in the tuple is inclusive (see examples below). 

        Returns:
        - pd.DataFrame: A subset of the DataFrame. Returns the full DataFrame if columns=None

        Examples:
        ```
        # Returns columns named 'column1' and 'column2':
        subset = df_info.get_slice(['column1', 'column2']) 

        # Returns column named 'column1' only
        subset = df_info.get_slice('column1') 

        # Returns the first three columns of the dataframe (column with index 2 is included)
        subset = df_info.get_slice((0,2)) 
        ```

        """
        # NOTE Like this slicing feature to analyse subsets of the data really nice
        if columns is not None:
            try:
                if isinstance(columns, list): # slice by column names
                    columns = [col.lower() for col in columns]
                    return self.df[columns]
                elif isinstance(columns, str): # choose one column only, allows for string outside list
                    columns = columns.lower()
                    return self.df[[columns]]
                elif isinstance(columns, tuple) and len(columns) == 2 and all(isinstance(col, int) for col in columns):
                    return self.df.iloc[:, columns[0]:columns[1] + 1]
                else:
                    raise ValueError
            # NOTE Great use of exception handling here very nice indeed
            except KeyError as ke:
                print(f"KeyError: you need to provide a valid column name: {ke}")
            except ValueError as ve: 
                print(f"ERROR: Invalid columns parameter. Use a list of valid column names formatted as strings, " \
                      + f"or a numerical interval formatted as a tuple e.g. (0,3): {ve}")
            except AttributeError as ae:
                print(f"ERROR: Invalid columns parameter. Use a list of valid column names formatted as strings, " \
                      f"or a numerical interval formatted as a tuple e.g. (0,3): {ae}")
        else:
            return self.df
        
    def extract_column_names(self, columns=None) -> List[str]:
        """
        Extract column names from a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset. 
        See docs for get_slice method for the requirements of the 'columns' parameter > help(get_slice)

        Returns:
        - List[str]: List of column names.

        Example:
        ```
        column_names = df_info.extract_column_names(['column1', 'column2'])
        ```

        """
        subset = self.get_slice(columns)
        return list(subset.columns)
    
    def data_types_columns(self, columns=None) -> pd.Series:
        """
        Get data types of columns in a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        See docs for get_slice method for the requirements of the 'columns' parameter > help(get_slice)

        Returns:
        - pd.Series: Data types of columns.

        Example:
        ```
        dtypes = df_info.data_types_columns(['column1', 'column2'])
        ```

        """
        subset = self.get_slice(columns)
        return subset.dtypes
    
    def info_columns(self, columns=None) -> None:
        """
        Display concise information about columns in a subset of the DataFrame: null counts and data types of columns.

        Parameters:
        - columns: Columns to include in the subset.
        See docs for get_slice method for the requirements of the 'columns' parameter > help(get_slice)

        Example:
        ```
        df_info.info_columns(['column1', 'column2'])
        ```

        """
        subset = self.get_slice(columns)
        return subset.info()
    
    def extract_statistical_values(self, columns=None) -> pd.DataFrame:
        """
        Extract statistical values from columns in a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        See docs for get_slice method for the requirements of the 'columns' parameter > help(get_slice)

        Returns:
        - pd.DataFrame: Statistical values of columns.

        Example:
        ```
        stats = df_info.extract_statistical_values(['column1', 'column2'])
        ```

        """
        subset = self.get_slice(columns)
        return subset.describe()
    
    def show_distinct_values(self,columns=None) -> None:
        """
        Display distinct values in columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        See docs for get_slice method for the requirements of the 'columns' parameter > help(get_slice)

        Example:
        ```
        df_info.show_distinct_values(['column1', 'column2'])
        ```

        """
        subset = self.get_slice(columns)
        for column in subset:
            try:
                print(f"Unique values in {column}:", np.sort(self.df[column].unique()))
            except TypeError:
                print(f"Unique values in {column}:", self.df[column].unique())

    def count_distinct_values(self, columns=None) -> pd.DataFrame:
        """
        Count distinct values in columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        See docs for get_slice method for the requirements of the 'columns' parameter > help(get_slice)

        Returns:
        - pd.DataFrame: Count of distinct values in columns.

        Example:
        ```
        counts = df_info.count_distinct_values(['column1', 'column2'])
        ```

        """
        subset = self.get_slice(columns)
        distinct_counts = pd.DataFrame({
            'column': subset.columns,
            'distinct_values_count': [subset[column].nunique() for column in subset.columns]
        })
        distinct_counts.set_index(['column'], inplace=True)
        return distinct_counts
    
    def print_shape(self, columns=None) -> None:
        """
        Print the shape of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        See docs for get_slice method for the requirements of the 'columns' parameter > help(get_slice)

        Example:
        ```
        df_info.print_shape(['column1', 'column2'])
        ```

        """
        subset = self.get_slice(columns)
        print("DataFrame Shape:", subset.shape)

    def generate_null_counts(self, columns=None) -> pd.DataFrame:
        """
        Generate null counts and percentages in columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        See docs for get_slice method for the requirements of the 'columns' parameter > help(get_slice)

        Returns:
        - pd.DataFrame: Null counts and percentages in columns.

        Example:
        ```
        null_info = df_info.generate_null_counts(['column1', 'column2'])
        ```

        """
        subset = self.get_slice(columns)
        null_counts = subset.isnull().sum()
        null_percentages = (null_counts / len(subset)) * 100
        null_info = pd.DataFrame({
            'null_count': null_counts,
            'null_percentage': null_percentages
        })
        return null_info
    
    def extract_numeric_features(self) -> pd.DataFrame:
        """
        Extract numeric features from the DataFrame.

        Returns:
        - pd.DataFrame: Numeric features.

        Example:
        ```
        numeric_features = df_info.extract_numeric_features()
        ```

        """
        numeric_features = self.df.select_dtypes(include=np.number)
        return numeric_features
    
    def extract_categorical_features(self, numeric_features: Optional[List[str]] = None) -> List[str]:
        """
        Extract categorical features from the DataFrame.

        Parameters:
        - numeric_features (Optional[List[str]]): List of numeric features (if already extracted).

        Returns:
        - List[str]: Categorical features.

        Example:
        ```
        categorical_features = df_info.extract_categorical_features(numeric_features=['num1', 'num2'])
        ```

        """
        if numeric_features is not None:
            categorical_features = [col for col in self.df.columns if col not in numeric_features]
            return categorical_features
        else:
            numeric_features = self.extract_numeric_features()
            categorical_features = [col for col in self.df.columns if col not in numeric_features]
            return categorical_features
        
    def print_summary_statistics(self, column_name: str) -> None:
        """
        Print summary statistics for a specific column.

        Parameters:
        - column_name (str): Name of the column.

        Example:
        ```
        df_info.print_summary_statistics('column1')
        ```

        """
        print(f"The mode of the distribution is {self.df[column_name].mode()[0]}")
        print(f"The mean of the distribution is {self.df[column_name].mean()}")
        print(f"The median of the distribution is {self.df[column_name].median()}")
    
    def data_skewness_values(self, columns: List[str]) -> None:
        """
        Calculate and display skewness values for specified columns in the DataFrame.

        Parameters:
        - columns (List[str]): List of column names to calculate skewness for.

        Returns:
        - None: The method prints the skewness values in a tabulated format.

        Example:
        ```
        your_instance.data_skewness_values(['column1', 'column2', 'column3'])
        ```

        """
        skew_data = []
        for col in columns:
            skew_value = self.df[col].skew()
            skew_data.append([col, skew_value])
        
        print(tabulate(skew_data, headers=[
              "Column", "Skewness"], tablefmt="pretty"))
