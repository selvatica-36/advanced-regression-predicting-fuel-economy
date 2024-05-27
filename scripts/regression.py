# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

# Import library modules
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as r2, mean_absolute_error as mae
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


class RegressionTools:
    def __init__(self):
        pass

    def cross_validation_OLS(self, X, y, n_splits):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=2023)
        
        # Create a list to store validation scores for each fold
        cv_lm_tr_r2s = []
        cv_lm_tr_mae = []
        cv_lm_val_r2s = []
        cv_lm_val_mae = []
        
        for train_ind, val_ind in kf.split(X, y):
            # Subset data based on CV folds
            X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
            X_val, y_val = X.iloc[val_ind], y.iloc[val_ind]
            # Fit the Model on fold's training data
            model = sm.OLS(y_train, X_train).fit()
            # Append Validation score to list 
            cv_lm_tr_r2s.append(r2(y_train, model.predict(X_train)))
            cv_lm_tr_mae.append(mae(y_train, model.predict(X_train)))
            cv_lm_val_r2s.append(r2(y_val, model.predict(X_val),))
            cv_lm_val_mae.append(mae(y_val, model.predict(X_val),))
            
            
        print(f"All Training R2s: {[round(x, 3) for x in cv_lm_tr_r2s]}")
        print(f"Training R2s: {round(np.mean(cv_lm_tr_r2s), 3)} +- {round(np.std(cv_lm_tr_r2s), 3)}")
        
        print(f"Training MAEs: {[round(x, 3) for x in cv_lm_tr_mae]}")
        print(f"Training MAEs: {round(np.mean(cv_lm_tr_mae), 3)} +- {round(np.std(cv_lm_tr_mae), 3)}")    
            
        print(f"All Validation R2s: {[round(x, 3) for x in cv_lm_val_r2s]}")
        print(f"Cross Val R2s: {round(np.mean(cv_lm_val_r2s), 3)} +- {round(np.std(cv_lm_val_r2s), 3)}")

        print(f"All Validation MAEs: {[round(x, 3) for x in cv_lm_val_mae]}")
        print(f"Cross Val MAEs: {round(np.mean(cv_lm_val_mae), 3)} +- {round(np.std(cv_lm_val_mae), 3)}")
        
        return model

    def residual_analysis_plots(self, model):
        predictions = model.predict()
        residuals = model.resid
        
        fig, ax = plt.subplots(1, 2, sharey="all", figsize=(10, 6))
        
        sns.scatterplot(x=predictions, y=residuals, ax=ax[0])
        ax[0].set_title("Residual Plot")
        ax[0].set_xlabel("Prediction")
        ax[0].set_ylabel("Residuals")
        
        stats.probplot(residuals, dist="norm", plot=ax[1])
        ax[1].set_title("Normal Q-Q Plot")   

    def VIF(self, X):
        '''
        Calculates VIF for all features (X) in the model
        '''
        VIF_series = pd.Series(
            [vif(X.values, i) for i in range(X.shape[1])],
            index=X.columns
        )
        return VIF_series