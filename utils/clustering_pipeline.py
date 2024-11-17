from sklearn.cluster import KMeans
from utils.custom_transformers import PipelineWithY

def create_clustering_pipeline(preprocessing_pipeline, n_clusters=3):
    clustering_pipeline = PipelineWithY(
        steps=[
            ('preprocessing', preprocessing_pipeline),
            ('clustering', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
        ]
    )
    return clustering_pipeline