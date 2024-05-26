from scripts.statistical_tests import StatisticalTests
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class OutlierDetector(StatisticalTests):
    """
    A class for performing outlier detection operations on a DataFrame.
    Inherits from StatisticalTests for statistical tests and data analysis capabilities.
    """
    def __init__(self, dataframe):
        super().__init__(dataframe)
    
    def z_scores(self, column: str) -> pd.DataFrame:
        """
        Calculate z-scores for a specific column.

        Z-scores measure the number of standard deviations a particular data point is from the mean.
        They are useful for identifying how extreme or unusual a data point is within a distribution.

        Parameters:
        - column (str): Name of the column to calculate z-scores.

        Returns:
        - pd.DataFrame: DataFrame with the original column values and corresponding z-scores.
        
        Example:
        ```
        outlier_detector = OutlierDetector(your_dataframe)
        z_scores_df = outlier_detector.z_scores('your_column')
        print(z_scores_df)
        ```
        """
        mean_col = np.mean(self.df[column])
        std_col = np.std(self.df[column])
        z_scores = (self.df[column] - mean_col) / std_col
        col_values = self.df[[column]].copy()
        col_values['z-scores'] = z_scores
        return col_values
    
    def IQR_outliers(self, column_list: List[str]) -> None:
        """
        Identify and print outliers using the Interquartile Range (IQR) method for multiple columns.

        Parameters:
        - column_list (List[str]): List of column names to identify outliers.

        """
        for col in column_list:
            Q1, Q3, IQR, results_str = self.IQR(col)
            outliers = self.df[(self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))]
            print("Outliers:")
            print(f'shape: {outliers.shape}')