from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_array

from typing import List, Union, Optional
import pandas as pd
import numpy as np


#-------- Defining the custom transformers --------#

class DropNaN(BaseEstimator, TransformerMixin):
    def __init__(self, columns_list: Optional[List[str]] = None, reset_index: bool = False) -> None:
        """
        Transformer that drops rows containing NaN values in specified columns or all columns.

        Parameters
        ----------
        columns_list : list of str, optional (default=None)
            List of column names to check for NaN values. If None, all columns are checked.
        reset_index : bool, optional (default=False)
            If True, resets the index of the returned DataFrame after dropping rows.
        """
        self.columns_list = columns_list
        self.reset_index = reset_index

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> 'DropNaN':
        # Validate input
        X = self._validate_input(X)

        # Store column names
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Removes rows containing missing values in the specified columns or in all columns if none are specified.

        Parameters
        ----------
        X : {array-like, DataFrame}, shape (n_samples, n_features)
            The data to clean.

        y : array-like, shape (n_samples,), optional (default=None)
            Target values (ignored in this transformer).

        Returns
        -------
        X_transformed : same as input type
            `X` with rows containing missing values removed.

        y_transformed : same as y
            `y` with corresponding rows removed if `y` is provided.
        """
        is_array = isinstance(X, np.ndarray)
        X = self._validate_input(X)

        # Check for consistency in columns
        if X.columns.tolist() != self.feature_names_in_:
            raise ValueError("The columns in the input data during transform differ from those during fit.")

        # Determine columns to check for NaN
        cols_to_check = self.columns_list if self.columns_list else X.columns

        # Check if specified columns exist
        missing_cols = [col for col in cols_to_check if col not in X.columns]
        if missing_cols:
            raise ValueError(f"The following columns were not found: {missing_cols}")

        # Drop rows with NaN values in specified columns
        if y is not None:
            y = pd.Series(y, name='target') if not isinstance(y, pd.Series) else y
            Xy = pd.concat([X, y], axis=1)
            Xy_transformed = Xy.dropna(subset=cols_to_check)
            X_transformed = Xy_transformed.drop(columns=[y.name])
            y_transformed = Xy_transformed[y.name]

            # Reset index if required
            if self.reset_index:
                X_transformed.reset_index(drop=True, inplace=True)
                y_transformed.reset_index(drop=True, inplace=True)

            # Convert back to original type if input was an array
            if is_array:
                X_transformed = X_transformed.values
                y_transformed = y_transformed.values

            return X_transformed, y_transformed
        else:
            X_transformed = X.dropna(subset=cols_to_check)

            # Reset index if required
            if self.reset_index:
                X_transformed.reset_index(drop=True, inplace=True)

            # Convert back to original type if input was an array
            if is_array:
                X_transformed = X_transformed.values

            return X_transformed

    def _validate_input(self, X):
        # Validate X and convert to DataFrame if necessary
        if isinstance(X, pd.DataFrame):
            return X.copy()
        elif isinstance(X, np.ndarray):
            X = check_array(X, ensure_2d=True, allow_nd=False, dtype=None)
            return pd.DataFrame(X, columns=getattr(self, 'feature_names_in_', None))
        else:
            raise TypeError("Input must be a pandas DataFrame or a NumPy array.")

    def get_feature_names_out(self, input_features=None) -> List[str]:
        # Return the feature names
        return self.feature_names_in_ if input_features is None else input_features
  
    
class SeniorStatusTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='customer_senior'):
        self.column = column

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].map({0: 'No', 1: 'Yes'}).astype('object')
        return X


class PipelineWithY(Pipeline):
    def fit(self, X, y=None, **fit_params):
        for name, transform in self.steps:
            if hasattr(transform, 'fit_transform'):
                # Attempt to fit_transform with both X and y
                try:
                    result = transform.fit_transform(X, y)
                    if isinstance(result, tuple) and len(result) == 2:
                        X, y = result
                    else:
                        X = result
                except TypeError:
                    # Transformer doesn't accept y
                    X = transform.fit_transform(X)
            else:
                X = transform.fit(X, y).transform(X)
        return self

    def transform(self, X, y=None):
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


def create_full_pipeline(df):
    #-------- Defining input variables --------#
    id_col = "id"
    target_col = "churn"
    senior_col = "customer_senior"

    drop_columns = ["phone_subscription", "streaming_tv"]
    excluded_columns = drop_columns + [id_col, target_col, senior_col]

    # Identifying categorical and numerical columns to process
    cat_columns = [senior_col] + [col for col in df.select_dtypes(include=['object']).columns if col not in excluded_columns]
    num_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col not in excluded_columns]

    #-------- Defining the transformers --------#
    cat_transformer = Pipeline(
        steps=[
            ('encoder', OneHotEncoder(sparse_output=False))
        ]
    )

    num_transformer = Pipeline(
        steps=[
            ('scaler', StandardScaler())
        ]
    )

    #-------- Defining the preparation pipeline --------#
    preparation = ColumnTransformer(
        transformers=[
            ('col_drop', 'drop', drop_columns),
            ('cat', cat_transformer, cat_columns),
            ('num', num_transformer, num_columns)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    #-------- Defining the full pipeline --------#
    full_pipeline = PipelineWithY(
        steps=[
            ('drop_nan', DropNaN(columns_list=None, reset_index=True)),
            ('senior_status', SeniorStatusTransformer(column=senior_col)),
            ('preparation', preparation)
        ]
    )

    return full_pipeline


if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    
    create_full_pipeline(df)