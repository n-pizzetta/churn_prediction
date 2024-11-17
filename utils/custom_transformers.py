from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

#############
## CLASSES ##
#############

#-------- Defining the custom transformers --------#
    
class SeniorStatusTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='customer_senior'):
        self.column = column

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].map({0: 'No', 1: 'Yes'}).astype('object')
        return X

class TargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='churn'):
        self.column = column

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, y):
        y = y.copy()
        # Transform the target values to 'No' and 'Yes'
        y = y.map({'No': 0, 'Yes': 1}).astype('float')
        return y

class FloatConverter(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_columns=None):
        self.exclude_columns = exclude_columns  # List of columns to exclude from conversion
        self.columns = None  # Placeholder for column names

    def fit(self, X, y=None):
        # Set self.columns to feature names
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()
        else:
            # If X is not a DataFrame, attempt to get feature names
            if hasattr(X, 'get_feature_names_out'):
                self.columns = X.get_feature_names_out()
            else:
                # If feature names are not available, create default names
                self.columns = [f'feature_{i}' for i in range(X.shape[1])]
        return self

    def transform(self, X):
        # Check if `columns` attribute has been set
        if not hasattr(self, 'columns') or self.columns is None:
            raise ValueError("Columns are not set in FloatConverter. Ensure columns are set during fit.")
        
        # Convert to DataFrame if X is an array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns)
        
        # Exclude specific columns and convert the rest to float
        if self.exclude_columns:
            cols_to_convert = [col for col in X.columns if col not in self.exclude_columns]
            X[cols_to_convert] = X[cols_to_convert].astype('float')
        else:
            X = X.astype('float')  # Convert all if no exclusions specified

        # Convert back to numpy array if necessary
        return X.values if not isinstance(X, pd.DataFrame) else X
        

class PipelineWithY(Pipeline):
    def __init__(self, steps, target_transformer=None):
        super().__init__(steps)
        self.target_transformer = target_transformer

    def fit(self, X, y=None, **fit_params):
        if y is not None and self.target_transformer:
            y = self.target_transformer.fit_transform(y)

        # Separate transformers and estimator
        transformers = self.steps[:-1]
        final_step_name, final_estimator = self.steps[-1]

        # Fit and transform with transformers
        for name, transform in transformers:
            if hasattr(transform, 'fit_transform'):
                X = transform.fit_transform(X, y)
            else:
                X = transform.fit(X, y).transform(X)

            # After 'preparation', get the feature names and set them in 'float_converter'
            if name == 'preparation':
                if hasattr(transform, 'get_feature_names_out'):
                    feature_names = transform.get_feature_names_out()
                    if 'float_converter' in dict(self.steps):
                        self.named_steps['float_converter'].columns = feature_names

        # Fit the final estimator
        if final_estimator is not None and final_estimator != 'passthrough':
            final_estimator.fit(X, y)
        return self

    def transform(self, X, y=None):
            if y is not None and self.target_transformer:
                y = self.target_transformer.transform(y)  # Apply y transformation in transform
            
            for name, transform in self.steps:
                if hasattr(transform, 'transform'):
                    try:
                        result = transform.transform(X, y)
                        if isinstance(result, tuple) and len(result) == 2:
                            X, y = result
                        else:
                            X = result
                    except TypeError:
                        # Transformer doesn't accept y
                        X = transform.transform(X)
            if y is not None:
                return X, y
            else:
                return X

    def predict(self, X):
        # Apply transforms sequentially
        for name, transform in self.steps[:-1]:
            X = transform.transform(X)
        # Predict with the final estimator
        final_step_name, final_estimator = self.steps[-1]
        return final_estimator.predict(X)

    def predict_proba(self, X):
        # Apply transforms sequentially
        for name, transform in self.steps[:-1]:
            X = transform.transform(X)
        # Predict probabilities with the final estimator
        final_step_name, final_estimator = self.steps[-1]
        if hasattr(final_estimator, 'predict_proba'):
            return final_estimator.predict_proba(X)
        else:
            raise AttributeError(
                f"Final estimator '{final_step_name}' does not support predict_proba."
            )
