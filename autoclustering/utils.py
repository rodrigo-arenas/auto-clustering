import numpy as np

from sklearn import metrics
from hdbscan.validity import validity_index


def score_candidate(pipe, X):
    """
    Scores the 'quality' of a clustering pipeline

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
    X : pd.DataFrame

    Returns
    -------
    score: dict
        Dict with popular clustering metrics
    """
    labels = pipe.named_steps["clustering"].labels_
    n_labels = len(np.unique(labels))
    if n_labels > 1:
        validity_score = validity_index(X, labels)
        silhouette_score = metrics.silhouette_score(X, labels, metric="euclidean")
        davies_bouldin = metrics.davies_bouldin_score(X, labels)
        calinski_harabasz = metrics.calinski_harabasz_score(X, labels)

    else:
        validity_score = -1000
        silhouette_score = -1000
        davies_bouldin = 1000
        calinski_harabasz = -1000

    return {"silhouette": silhouette_score,
            "davies_bouldin": davies_bouldin,
            "validity_index": validity_score,
            "calinski_harabasz": calinski_harabasz}
