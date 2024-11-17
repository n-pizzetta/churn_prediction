import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils.custom_transformers import (
    SeniorStatusTransformer,
    FloatConverter
)

def create_preprocessing_pipeline(df):
    # Define input variables
    id_col = "id"
    target_col = "churn"
    senior_col = "customer_senior"

    drop_columns = ["amount_total_charges", "phone_subscription", "streaming_tv"]
    excluded_columns = drop_columns + [id_col, target_col, senior_col]

    # Identify categorical and numerical columns to process
    cat_columns = [senior_col] + [col for col in df.select_dtypes(include=['object']).columns if col not in excluded_columns]
    num_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col not in excluded_columns]

    # Define imputers
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Define transformers
    cat_transformer = Pipeline(
        steps=[
            ('imputer', cat_imputer),
            ('encoder', OneHotEncoder(drop="first", sparse_output=False))
        ]
    )

    num_transformer = Pipeline(
        steps=[
            ('imputer', num_imputer),
            ('scaler', StandardScaler())
        ]
    )

    # Define preparation pipeline
    preparation = ColumnTransformer(
        transformers=[
            ('col_drop', 'drop', drop_columns),
            ('cat', cat_transformer, cat_columns),
            ('num', num_transformer, num_columns)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    # Define preprocessing pipeline
    preprocessing_pipeline = Pipeline(
        steps=[
            ('senior_status', SeniorStatusTransformer(column=senior_col)),
            ('preparation', preparation),
            ('float_converter', FloatConverter(exclude_columns=[id_col]))
        ]
    )

    return preprocessing_pipeline