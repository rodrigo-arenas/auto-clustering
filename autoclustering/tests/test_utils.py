from .. import AutoClustering
from .. import score_candidate
from sklearn.datasets import load_digits

X, _ = load_digits(return_X_y=True)

clustering = AutoClustering(num_samples=3,
                            metric='validity_index',
                            verbose=0)
clustering.fit(X)


def test_score_candidate_result():
    scores = score_candidate(pipe=clustering.best_estimator_, X=X)
    metrics = ["silhouette", "davies_bouldin", "validity_index", "calinski_harabasz"]
    for metric in metrics:
        assert metric in scores.keys()


def test_wrong_metrics():
    penalization = 500

    scores = score_candidate(pipe=X, X=X, penalization=penalization)

    assert scores["silhouette"] == -penalization
    assert scores["validity_index"] == -penalization
    assert scores["davies_bouldin"] == penalization
    assert scores["calinski_harabasz"] == -penalization
