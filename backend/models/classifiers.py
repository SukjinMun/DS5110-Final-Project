"""
Classification models for ESI level prediction

Implements logistic regression, LDA, and Naive Bayes classifiers
for predicting Emergency Severity Index (ESI) levels.

Based on Ch4 Classification from DS 5110 course materials.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

class ESIClassifier:
    """
    Classifier for predicting ESI levels (1-5) based on patient data.

    Features:
    - Chief complaint (categorical)
    - Vital signs (heart rate, BP, respiratory rate, temperature, SpO2, pain score)
    - Arrival mode (categorical)
    - Patient demographics (age, sex)
    - Payor type (categorical)
    """

    def __init__(self, model_type='logistic'):
        """
        Initialize classifier

        Args:
            model_type: 'logistic', 'lda', or 'naive_bayes'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

        if model_type == 'logistic':
            self.model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        elif model_type == 'lda':
            self.model = LinearDiscriminantAnalysis()
        elif model_type == 'naive_bayes':
            self.model = GaussianNB()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def prepare_features(self, df):
        """
        Prepare features from raw encounter data

        Args:
            df: DataFrame with encounter data including vitals

        Returns:
            X: Feature matrix
            y: ESI level labels (if present)
        """
        # NOTE: Feature engineering is performed in scripts/train_models.py
        # This method is kept for API compatibility but not currently used
        pass

    def train(self, X_train, y_train):
        """
        Train the classifier

        Args:
            X_train: Training features
            y_train: Training labels (ESI levels 1-5)
        """
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()

        return self

    def predict(self, X):
        """
        Predict ESI levels

        Args:
            X: Feature matrix

        Returns:
            Predicted ESI levels
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """
        Predict class probabilities

        Args:
            X: Feature matrix

        Returns:
            Probability matrix (n_samples, 5) for ESI levels 1-5
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        return results

    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation

        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds

        Returns:
            Cross-validation scores
        """
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv)

        return {
            'cv_scores': scores.tolist(),
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std()
        }

    def save(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_names': self.feature_names
            }, f)

    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        classifier = cls(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        classifier.feature_names = data['feature_names']

        return classifier


def compare_classifiers(X_train, X_test, y_train, y_test):
    """
    Compare performance of different classification algorithms

    Returns:
        Dictionary with results for each classifier
    """
    results = {}

    for model_type in ['logistic', 'lda', 'naive_bayes']:
        clf = ESIClassifier(model_type=model_type)
        clf.train(X_train, y_train)
        eval_results = clf.evaluate(X_test, y_test)

        results[model_type] = {
            'accuracy': eval_results['accuracy'],
            'confusion_matrix': eval_results['confusion_matrix']
        }

    return results
