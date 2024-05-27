from scripts.info_extractor import DataFrameInfo
from scipy.stats import chi2, chi2_contingency, normaltest
from typing import List, Tuple
import math
import numpy as np
import pandas as pd


class StatisticalTests(DataFrameInfo):
    """
    A class for performing statistical tests on a DataFrame.
    Inherits from DataFrameInfo for additional data analysis capabilities.
    """

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    # NOTE Really like the method though 
    def chi_square_test(self, independent_variable: str, dependent_variables: List[str]) -> float:
        """
        Perform chi-square test between two categorical variables.

        The chi-square test is used to determine if there is a significant association
        between two categorical variables. It is useful for understanding whether the
        presence or absence of one variable is related to the presence or absence of another.

        Parameters:
        - independent_variable (str): Name of the independent variable.
        - dependent_variables (List[str]): List of dependent variables to test against the independent variable.

        Returns:
        - float: p-value of the chi-square test.

        """
        # Only between categorical variables
        chi_sq_test_df = self.df.copy()
        chi_sq_test_df[independent_variable] = chi_sq_test_df[independent_variable].isnull()
        # Step 2: Crosstab the new column with B
        if len(dependent_variables) > 3:
            for column in dependent_variables:
                contingency_table = pd.crosstab(chi_sq_test_df[independent_variable], chi_sq_test_df[column])
                # Step 3: Perform chi-squared test
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                if p < 0.05:
                    print(f"Chi-square test for missing values in {independent_variable} against {column} column: ")
                    print(f"p-value = {p}: Significant")
                    return p
                elif math.isclose(p, 0.05):
                    print(f"Chi-square test for missing values in {independent_variable} against {column} column: ")
                    print(f"p-value = {p}: Likely not significant")
                    return p
                
        elif len(dependent_variables) <= 3:
            for column in dependent_variables:
                contingency_table = pd.crosstab(chi_sq_test_df[independent_variable], chi_sq_test_df[column])
                # Step 3: Perform chi-squared test
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                print(f"Chi-square test for missing values in {independent_variable} against {column} column: ")
                print(f"p-value = {p}")
                return p
            
    def agostino_K2_test(self, column_name: str) -> None:
        """
        Perform D'Agostino's K^2 normality test on a continuous variable.

        D'Agostino's K^2 test is used to assess whether a sample comes from a specific distribution,
        in this case, the normal distribution. It is particularly useful for checking the normality
        assumption in statistical analyses.

        Parameters:
        - column_name (str): Name of the continuous variable to test.
        """
        # Test for normality in continuous variables
        stat, p = normaltest(self.df[column_name], nan_policy='omit')
        print('Statistics=%.3f, p=%.3f' % (stat, p))

    def IQR(self, column: str) -> Tuple[float, float, float, str]:
        """
        Calculate and display Interquartile Range (IQR) statistics for a single column.

        The Interquartile Range (IQR) is a measure of statistical dispersion, representing
        the range between the first quartile (Q1) and the third quartile (Q3). It is useful
        for identifying the spread of values in a dataset and detecting potential outliers.

        Parameters:
        - column (str): Name of the column to calculate IQR.

        Returns:
        - Tuple[float, float, float, str]: Tuple containing Q1, Q3, IQR, and a string with printed information.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        result_str = f"\nResults for {column} column:"
        result_str += f"\nQ1 (25th percentile): {Q1}"
        result_str += f"\nQ3 (75th percentile): {Q3}"
        result_str += f"\nIQR: {IQR}\n"
        print(result_str)
        return Q1, Q3, IQR, result_str
    
    def IQR_multiple_columns(self, column_list: List[str]) -> List[Tuple[float, float, float, str]]:
        """
        Calculate and display Interquartile Range (IQR) statistics for multiple columns.

        This method iterates over a list of continuous variables and calculates the
        Interquartile Range (IQR) for each, providing insights into the spread of values
        and potential outliers in multiple columns.

        Parameters:
        - column_list (List[str]): List of column names to calculate IQR for.

        Returns:
        - List[Tuple[float, float, float, str]]: List of tuples, each containing Q1, Q3, IQR, 
          and a string with printed information.
        """
        results = []
        for col in column_list:
            result_tuple = self.IQR(col)
            results.append(result_tuple)
            print(result_tuple[-1])  # Print the string information
        return results
    
    