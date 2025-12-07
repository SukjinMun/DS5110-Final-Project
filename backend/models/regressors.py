"""
Regression models for wait time prediction

Implements linear regression and Poisson GLM for predicting
ED wait times and patient volumes.

Based on Ch3 Linear Regression from DS 5110 course materials.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson
import pickle

class WaitTimePredictor:
    """
    Linear regression model for predicting wait times

    Target: Wait time (minutes) from arrival to provider

    Features:
    - ESI level (1-5)
    - Arrival mode (Walk-in, EMS, Transfer)
    - Chief complaint
    - Vital signs (heart rate, BP, etc.)
    - Time of day / day of week
    - Current ED volume
    """

    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.statsmodels_result = None

    def prepare_features(self, df):
        """
        Prepare features from encounter data

        NOTE: Wait time calculation requires proper datetime parsing.
        This will be fully implemented after date format is fixed.

        Args:
            df: DataFrame with encounter data

        Returns:
            X: Feature matrix
            y: Wait times in minutes
        """
        # NOTE: Feature engineering is performed in scripts/train_models.py
        # This method is kept for API compatibility but not currently used
        pass

    def train(self, X_train, y_train):
        """
        Train linear regression model

        Args:
            X_train: Training features
            y_train: Training wait times (minutes)
        """
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train sklearn model
        self.model.fit(X_train_scaled, y_train)

        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()

        return self

    def train_with_statsmodels(self, X_train, y_train):
        """
        Train using statsmodels for detailed statistical output

        Args:
            X_train: Training features
            y_train: Training wait times
        """
        # Add constant
        X_train_const = sm.add_constant(X_train)

        # Fit OLS model
        self.statsmodels_result = sm.OLS(y_train, X_train_const).fit()

        return self.statsmodels_result

    def predict(self, X):
        """
        Predict wait times

        Args:
            X: Feature matrix

        Returns:
            Predicted wait times in minutes
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Actual wait times

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)

        results = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'coefficients': self.model.coef_.tolist() if self.model.coef_ is not None else None
        }

        if self.feature_names:
            results['feature_importance'] = dict(zip(self.feature_names, self.model.coef_))

        return results

    def get_summary(self):
        """
        Get detailed statistical summary from statsmodels

        Returns:
            Summary table with coefficients, p-values, R-squared, etc.
        """
        if self.statsmodels_result:
            return str(self.statsmodels_result.summary())
        else:
            return "No statsmodels result available. Run train_with_statsmodels() first."

    def save(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)

    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        predictor = cls()
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_names = data['feature_names']

        return predictor


class PatientVolumePredictor:
    """
    Poisson GLM for predicting patient arrival volumes

    Target: Number of patients arriving per hour

    Features:
    - Hour of day
    - Day of week
    - Month
    - Holiday indicator
    """

    def __init__(self):
        self.model = None
        self.feature_names = None

    def prepare_time_series(self, df):
        """
        Prepare time series data for volume prediction

        NOTE: Requires proper datetime parsing.
        Will be fully implemented after date format is fixed.

        Args:
            df: DataFrame with encounter data

        Returns:
            Time series with hourly patient counts
        """
        # NOTE: Time series preparation is performed in scripts/train_models.py
        # This method is kept for API compatibility but not currently used
        pass

    def train(self, X_train, y_train):
        """
        Train Poisson GLM

        Args:
            X_train: Training features (temporal features)
            y_train: Training target (patient counts per hour)
        """
        # Add constant
        X_train_const = sm.add_constant(X_train)

        # Fit Poisson GLM
        self.model = sm.GLM(y_train, X_train_const, family=Poisson()).fit()

        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()

        return self

    def predict(self, X):
        """
        Predict patient volumes

        Args:
            X: Feature matrix

        Returns:
            Predicted patient counts
        """
        X_const = sm.add_constant(X)
        return self.model.predict(X_const)

    def evaluate(self, X_test, y_test):
        """
        Evaluate Poisson model performance

        Args:
            X_test: Test features
            y_test: Actual patient counts

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)

        results = {
            'deviance': self.model.deviance,
            'aic': self.model.aic,
            'bic': self.model.bic,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }

        return results

    def get_summary(self):
        """Get detailed statistical summary"""
        if self.model:
            return str(self.model.summary())
        else:
            return "No model available. Run train() first."
