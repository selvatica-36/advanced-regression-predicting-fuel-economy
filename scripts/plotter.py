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

    def correlation_heatmap(self, column_list: List[str]) -> None:
        """
        Creates a heatmap of the correlation matrix.

        Parameters:
        - column_list (List[str]): List of column names for correlation analysis.

        """
        sns.heatmap(self.df[column_list].corr(), annot=True, cmap='coolwarm')

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
                    square=True, linewidths=.5, annot=True, cmap=cmap)
        plt.yticks(rotation=0)
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.show()

    def qq_plot(self, column_list: List[str]) -> None:
        """
        Creates QQ plots for a list of columns.

        Parameters:
        - column_list (List[str]): List of column names for QQ plot.

        """
        for column in column_list:
            qqplot(self.df[column], scale=1 ,line='q')

    def multi_qq_plot(self, columns: List[str]) -> None:
        """
        Creates multiple QQ plots in a grid.

        Parameters:
        - columns (List[str]): List of column names for QQ plot.

        """
        remainder = 1 if len(columns) % 4 != 0 else 0
        rows = int(len(columns) / 4 + remainder)
        fig, axes = plt.subplots(
            ncols=3, nrows=rows, sharex=False, figsize=(12, 6))
        # np.ravel flattens the 2d axis array, meaning that we iterate and plot on x:y axis
        for col, ax in zip(columns, np.ravel(axes)):
            qqplot(self.df[col], line='s', ax=ax, fit=True)
            ax.set_title(f'{col} QQ Plot')
        plt.tight_layout()

    def nulls_dataframe_plot(self) -> None:
        """
        Creates a matrix plot showing the nullity of the DataFrame.

        """
        msno.matrix(self.df)

    def pair_correlations_grid(self, numeric_features: List[str] = None) -> None:
        """
        Creates a pair plot for numerical features.

        Parameters:
        - numeric_features (List[str], optional): List of numerical feature names.

        """
        if numeric_features is not None:
            sns.pairplot(self.df[numeric_features])
        else:
            numeric_features = super().extract_numeric_features()
            sns.pairplot(self.df[numeric_features])

    def numeric_distributions_grid(self, numeric_features: List[str]=None, kde: bool=True) -> None:
        """
        Creates a grid of histograms for numerical features.

        Parameters:
        - numeric_features (List[str], optional): List of numerical feature names.
        - kde (bool, optional): Whether to include kernel density estimate. Default is True.

        """
        if numeric_features is not None:
            sns.set(font_scale=0.7)
            f = pd.melt(self.df, value_vars=numeric_features)
            g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
            g = g.map(sns.histplot, "value", kde=kde)
        else:
            numeric_features = super().extract_numeric_features()
            sns.set(font_scale=0.7)
            f = pd.melt(self.df, value_vars=numeric_features)
            g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
            g = g.map(sns.histplot, "value", kde=kde)

    def count_plot(self, x: str, **kwargs) -> None:
        """
        Creates a count plot for a categorical variable.

        Parameters:
        - x (str): Name of the categorical variable.
        - **kwargs: Additional keyword arguments for Seaborn's countplot.

        """
        sns.countplot(x=x)
        x=plt.xticks(rotation=90)

    def count_plots_grid(self, categorical_features: List[str]) -> None:
        """
        Creates a grid of count plots for categorical features.

        Parameters:
        - categorical_features (List[str]): List of categorical feature names.

        """
        figure = pd.melt(self.df, value_vars=categorical_features)
        grid = sns.FacetGrid(figure, col='variable',  col_wrap=3, sharex=False, sharey=False)
        grid = grid.map(self.count_plot, 'value')

    def log_transform_plot(self, col: str) -> None:
        """
        Creates a histogram for log-transformed values.

        Parameters:
        - col (str): Name of the column to be log-transformed.

        """
        nonzero_values = self.df[col][self.df[col] != 0]
        log_col = nonzero_values.map(lambda i: np.log(i) if i > 0 else 0)
        figure = sns.histplot(log_col,label="Skewness: %.2f"%(log_col.skew()), kde=True)
        figure.legend()
    
    def boxcox_transform_plot(self, col: str) -> None:
        """
        Creates a histogram for Box-Cox transformed values.

        Parameters:
        - col (str): Name of the column to be Box-Cox transformed.

        """
        boxcox_population = self.df[col]
        boxcox_population= stats.boxcox(boxcox_population)
        boxcox_population= pd.Series(boxcox_population[0])
        figure = sns.histplot(boxcox_population,label="Skewness: %.2f"%(boxcox_population.skew()))
        figure.legend()
    
    def yeo_johnson_transform_plot(self, col: str) -> None:
        """
        Creates a histogram for Yeo-Johnson transformed values.

        Parameters:
        - col (str): Name of the column to be Yeo-Johnson transformed.

        """
        nonzero_values = self.df[col][self.df[col] != 0]
        yeojohnson_population = stats.yeojohnson(nonzero_values)
        yeojohnson_population= pd.Series(yeojohnson_population[0])
        figure = sns.histplot(yeojohnson_population,label="Skewness: %.2f"%(yeojohnson_population.skew()))
        figure.legend()