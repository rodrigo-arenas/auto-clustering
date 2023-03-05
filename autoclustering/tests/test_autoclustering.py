import pytest
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from ray import tune
from sklearn.cluster import DBSCAN, AffinityPropagation
from hdbscan import HDBSCAN

from .. import AutoClustering

X, _ = load_iris(as_frame=True, return_X_y=True)
pipe = AutoClustering(num_samples=3, metric="davies_bouldin", max_concurrent_trials=1)
pipe.fit(X)


def test_expected_properties():
    assert isinstance(pipe.best_params_, dict)
    assert isinstance(pipe.best_estimator_, Pipeline)


def test_expected_output():
    labels = pipe.fit_predict(X)
    assert len(labels) == X.shape[0]
    assert pipe.n_clusters_ >= 1


def test_wrong_metric():
    model = AutoClustering(metric="accuracy")
    with pytest.raises(Exception) as excinfo:
        model.fit(X)

    assert str(excinfo.value) == f"Metric must be " \
                                 f"one of ['validity_index', 'davies_bouldin', 'silhouette', 'calinski_harabasz'], " \
                                 f"but got accuracy instead."


def test_n_jobs_parameter():
    n_jobs_list = [1, 5, -2, -1]
    for n_jobs in n_jobs_list:
        model = AutoClustering(n_jobs=n_jobs)
        assert model.n_jobs == n_jobs

    model = AutoClustering(n_jobs=0)
    assert model.n_jobs == -1

    with pytest.warns(UserWarning) as record:
        AutoClustering(n_jobs=-5)

    assert record[0].message.args[0] == "`self.n_jobs` is automatically set -1 for any negative values."


@pytest.mark.parametrize(
    "config",
    [
        ({"model": DBSCAN(), "params": {"eps": tune.loguniform(0.5, 4),
                                        "min_samples": tune.randint(5, 20)}}),
        ({"model": HDBSCAN(), "params": {"min_cluster_size": tune.randint(5, 30),
                                         "cluster_selection_method": tune.choice(["eom", "leaf"])}}),
        ({"model": AffinityPropagation(), "params": {"damping": tune.uniform(0.5, 0.99),
                                                     "convergence_iter": tune.randint(5, 20),
                                                     "max_iter": tune.randint(150, 300)}})

    ]
)
def test_custom_models(config):
    model_config = [config]
    custom_pipe = AutoClustering(num_samples=3,
                                 clustering_models=model_config,
                                 max_concurrent_trials=1)
    custom_pipe.fit(X)

    labels = custom_pipe.fit_predict(X)
    assert len(labels) == X.shape[0]

