import numpy as np
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from sklearn import metrics
from .configs import clustering_config, preprocessing_config
from .pipelines import get_pipelines


class AutoClustering:
    def __init__(self, alpha: float = 0.5, num_samples: int = 25, metric: str = 'multimetric'):
        self.alpha = alpha
        self.num_samples = num_samples
        self.metric = metric

    def fit(self, X):
        pipelines = get_pipelines(preprocessing_config, clustering_config)

        results = []
        for pipe in pipelines:
            tuner = tune.Tuner(tune.with_parameters(self.train, pipe=pipe["pipe"], X=X),
                               param_space=pipe["params"],
                               tune_config=tune.TuneConfig(
                                   search_alg=OptunaSearch(),
                                   metric=self.metric,
                                   mode=self.optimization_mode_,
                                   num_samples=self.num_samples)
                               )
            trial_result = tuner.fit()
            best_result = trial_result.get_best_result(metric=self.metric, mode=self.optimization_mode_)
            results.append(best_result)

        return results

    def train(self, config, pipe, X):
        pipe.set_params(**config)
        pipe.fit(X)
        scores = self.score(pipe, X)
        tune.report(**scores)

    def score(self, pipe, X):
        labels = pipe[-1].labels_
        n_labels = len(np.unique(labels))
        if n_labels > 1:
            silhouette_score = metrics.silhouette_score(X, labels, metric="euclidean")
            davies_bouldin = metrics.davies_bouldin_score(X, labels)

        else:
            silhouette_score = -1000
            davies_bouldin = 1000

        multimetric = self.alpha * (1 - silhouette_score) + (1 - self.alpha) * davies_bouldin

        return {"silhouette": silhouette_score, "davies_bouldin": davies_bouldin, "multimetric": multimetric}

    @property
    def optimization_mode_(self):
        modes_map = {"multimetric": 'min', "davies_bouldin": 'min', "silhouette": 'max'}
        metric_mode = modes_map.get(self.metric)
        if not metric_mode:
            raise ValueError(f"Metric must be one of ['multimetric', 'davies_bouldin', 'silhouette'], "
                             f"but got {self.metric} instead")

        return metric_mode
