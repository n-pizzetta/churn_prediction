import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
# Add the root directory to the path
sys.path.append(os.path.abspath(".."))
from utils.preprocessing_pipeline import create_preprocessing_pipeline
from utils.clustering_pipeline import create_clustering_pipeline
from utils.classification_pipeline import create_classification_pipeline
from utils.custom_transformers import TargetTransformer

# Load data
df = pd.read_csv("../data/data.csv")

# Create preprocessing pipeline
preprocessing_pipeline = create_preprocessing_pipeline(df)

# ---- Clustering ----
# Prepare data for clustering (exclude 'id' and 'churn')
X_clustering = df.drop(columns=['id', 'churn'])

# Create and fit clustering pipeline
clustering_pipeline = create_clustering_pipeline(preprocessing_pipeline, n_clusters=3)
clustering_pipeline.fit(X_clustering)

# Save clustering pipeline
joblib.dump(clustering_pipeline, '../models/clustering_pipeline.joblib')

# Add cluster labels to the dataframe
df['cluster'] = clustering_pipeline.named_steps['clustering'].labels_

# ---- Classification ----
# Prepare data for classification
X = df.drop(columns=['churn'])
y = df['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Exclude 'id' from features
X_train_features = X_train.drop(columns=['id'])
X_test_features = X_test.drop(columns=['id'])

# Transform target variable
target_transformer = TargetTransformer(column='churn')
#y_train_transformed = target_transformer.fit_transform(y_train)
#y_test_transformed = target_transformer.transform(y_test)

# Create classification pipeline
classification_pipeline = create_classification_pipeline(preprocessing_pipeline, target_transformer)

# Fit classification pipeline
classification_pipeline.fit(X_train_features, y_train)

# Save classification pipeline and target transformer
joblib.dump(classification_pipeline, '../models/classification_pipeline.joblib')
joblib.dump(target_transformer, '../models/target_transformer.joblib')
