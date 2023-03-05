from pandas.errors import InvalidIndexError

from sklearn import metrics
from hdbscan.validity import validity_index


def score_candidate(pipe, X, penalization=1000):
    """
    Scores the 'quality' of a clustering pipeline

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        pipeline to evaluate the results
    X : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
    penalization : int, default=1000
        penalization value to the score when not clusters found

    Returns
    -------
    score: dict
        Dict with popular clustering metrics
    """

    try:
        labels = pipe.named_steps["clustering"].labels_
        validity_score = validity_index(X, labels)
        silhouette_score = metrics.silhouette_score(X, labels, metric="euclidean")
        davies_bouldin = metrics.davies_bouldin_score(X, labels)
        calinski_harabasz = metrics.calinski_harabasz_score(X, labels)

    except (InvalidIndexError, ValueError, AttributeError):
        # Penalize algorithms with one cluster or with distance errors
        validity_score = -penalization
        silhouette_score = -penalization
        davies_bouldin = penalization
        calinski_harabasz = -penalization

    return {"silhouette": silhouette_score,
            "davies_bouldin": davies_bouldin,
            "validity_index": validity_score,
            "calinski_harabasz": calinski_harabasz}
