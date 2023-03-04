import numpy as np
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from ray.air import RunConfig
from sklearn import metrics
from hdbscan.validity import validity_index
from .configs import clustering_config, preprocessing_config
from .pipelines import get_pipelines


class AutoClustering:
    def __init__(self,
                 num_samples: int = 25,
                 metric: str = 'validity_index',
                 verbose=0,
                 n_jobs=1):
        self.num_samples = num_samples
        self.metric = metric
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X):
        search_space = get_pipelines(preprocessing_config, clustering_config)

        config = {"algorithm": tune.choice(list(search_space.keys())),
                  "search_space": search_space}

        tuner = tune.Tuner(tune.with_parameters(self.train, X=X),
                           param_space=config,
                           tune_config=tune.TuneConfig(
                               search_alg=OptunaSearch(),
                               max_concurrent_trials=self.n_jobs,
                               metric=self.metric,
                               mode=self.optimization_mode_,
                               num_samples=self.num_samples),
                           run_config=RunConfig(verbose=self.verbose)
                           )
        trial_result = tuner.fit()

        best_result = trial_result.get_best_result(metric=self.metric, mode=self.optimization_mode_)

        # Refit best pipeline
        best_pipeline = best_result.config["algorithm"]
        best_model = best_result.config["search_space"][best_pipeline]

        self.best_estimator_ = best_model["pipe"]

        self.best_params_ = best_model["params"]
        self.best_estimator_.set_params(**self.best_params_).fit(X)
        self.best_score_ = best_result.metrics[self.metric]

        return self

    def train(self, config, X):
        algorithm = config["algorithm"]
        pipeline_dict = config["search_space"][algorithm]

        pipe = pipeline_dict["pipe"]
        params = pipeline_dict["params"]
        pipe.set_params(**params)
        pipe.fit(X)
        scores = self.score(pipe, X)
        tune.report(**scores)

    def score(self, pipe, X):
        labels = pipe.named_steps["clustering"].labels_
        n_labels = len(np.unique(labels))
        if n_labels > 1:
            validity_score = validity_index(X, labels)
            silhouette_score = metrics.silhouette_score(X, labels, metric="euclidean")
            davies_bouldin = metrics.davies_bouldin_score(X, labels)

        else:
            validity_score = -1000
            silhouette_score = -1000
            davies_bouldin = 1000

        return {"silhouette": silhouette_score,
                "davies_bouldin": davies_bouldin,
                "validity_index": validity_score}

    @property
    def optimization_mode_(self):
        modes_map = {"validity_index": 'max', "davies_bouldin": 'min', "silhouette": 'max'}
        metric_mode = modes_map.get(self.metric)
        if not metric_mode:
            raise ValueError(f"Metric must be one of ['validity_index', 'davies_bouldin', 'silhouette'], "
                             f"but got {self.metric} instead")

        return metric_mode
