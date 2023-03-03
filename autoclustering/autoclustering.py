import numpy as np
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from sklearn import metrics
from .configs import clustering_config, preprocessing_config
from .pipelines import get_pipelines


class AutoClustering:
    def __init__(self, alpha: float = 0.5, num_samples=25):
        self.alpha = alpha
        self.num_samples = num_samples

    def fit(self, X):
        pipelines = get_pipelines(preprocessing_config, clustering_config)

        results = []
        for pipe in pipelines:
            tuner = tune.Tuner(tune.with_parameters(self.train, pipe=pipe["pipe"], X=X),
                               param_space=pipe["params"],
                               tune_config=tune.TuneConfig(
                                   search_alg=OptunaSearch(metric=["silhouette", "davies_bouldin", "score"],
                                                           mode=["max", "min", "min"]),
                                   num_samples=self.num_samples)
                               )
            trial_result = tuner.fit()
            best_result = trial_result.get_best_result(metric="score", mode="min")
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

        score = self.alpha*(1-silhouette_score) + (1-self.alpha)*davies_bouldin

        return {"silhouette": silhouette_score, "davies_bouldin": davies_bouldin, "score": score}
