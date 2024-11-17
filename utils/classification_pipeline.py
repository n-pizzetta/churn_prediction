from sklearn.linear_model import LogisticRegression
from utils.custom_transformers import PipelineWithY

def create_classification_pipeline(preprocessing_pipeline, target_transformer):
    classification_pipeline = PipelineWithY(
        steps=[
            ('preprocessing', preprocessing_pipeline),
            ('logistic_regression', LogisticRegression(
                solver='liblinear',
                C=10,
                penalty='l2',
                random_state=42
            ))
        ],
        target_transformer=target_transformer
    )
    return classification_pipeline
