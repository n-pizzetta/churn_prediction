import joblib
import pandas as pd

# Load models
clustering_pipeline = joblib.load('../models/clustering_pipeline.joblib')
classification_pipeline = joblib.load('../models/classification_pipeline.joblib')
target_transformer = joblib.load('../models/target_transformer.joblib')

# Load new data (could be new clients or the test set)
new_data = pd.read_csv('../data/new_clients.csv')

# ---- Predict Clusters ----
# Prepare data (exclude 'id' and 'churn' if present)
X_new = new_data.drop(columns=['id', 'churn'], errors='ignore')

# Predict clusters
clusters = clustering_pipeline.predict(X_new)

# Add clusters to new_data
new_data['cluster'] = clusters

# ---- Predict Churn ----
# Exclude 'id' from features
X_new_features = new_data.drop(columns=['id'])

# Predict churn probabilities
churn_proba = classification_pipeline.predict_proba(X_new_features)[:, 1]

# Add churn probabilities to new_data
new_data['predicted_churn_proba'] = churn_proba

# If you need predicted classes
churn_pred = classification_pipeline.predict(X_new_features)
new_data['predicted_churn'] = churn_pred

# If necessary, inverse transform the target variable (e.g., map numeric labels back to original)
# Assuming target_transformer has an inverse_transform method
# new_data['predicted_churn'] = target_transformer.inverse_transform(churn_pred)

# Save predictions
new_data.to_csv('../data/new_clients_with_predictions.csv', index=False)
