from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from scripts.statistical_tests import StatisticalTests
from typing import List
import missingno as msno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter(StatisticalTests):
    """
    A class for creating various plots and visualizations based on a DataFrame.
    """

    def __init__(self, dataframe):
        self.df = dataframe.copy()
        
    def discrete_probability_distribution(self, column_name: str, **kwargs) -> None:
        """
        Creates a bar plot for discrete probability distribution.

        Parameters:
        - column_name (str): Name of the column for which to create the plot.
        - **kwargs: Additional keyword arguments for Seaborn's barplot.

        """
        plt.rc("axes.spines", top=False, right=False)
        sns.set_style(style='darkgrid', rc=None)
        probs = self.df[column_name].value_counts(normalize=True)
        # Create bar plot
        dpd = sns.barplot(y=probs.values, x=probs.index)
        dpd.set_xticklabels(dpd.get_xticklabels(), rotation=45, ha='right')
        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.title('Discrete Probability Distribution')
        plt.show()

    def continuous_probability_distribution(self, column_name: str, column_list: List[str] = None) -> None:
        """
        Creates a histogram for continuous probability distribution.

        Parameters:
        - column_name (str): Name of the column for which to create the plot.
        - column_list (List[str], optional): List of additional columns for comparison.
        
        """
        if column_list is not None:
            for col in column_list:
                sns.histplot(self.df[col], kde=True, color='blue', stat="probability", bins=30)
                super().print_summary_statistics(col)
        else:
            sns.histplot(self.df[column_name], kde=True, color='blue', stat="probability", bins=30)
            super().print_summary_statistics(column_name)

    def correlation_matrix_df(self) -> None:
        """
        Creates a heatmap of the correlation matrix for all numerical variables.

        """
        corr = self.df.select_dtypes(include=np.number).corr()
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap
        sns.heatmap(corr, mask=mask, 
                    square=True, linewidths=.5, annot=True, cmap=cmap, vmin=-1, vmax=1)
        plt.yticks(rotation=0)
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.show()