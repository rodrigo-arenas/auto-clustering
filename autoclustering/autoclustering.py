import numpy as np
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from hdbscan import HDBSCAN

from ray.air import RunConfig
from .configs import clustering_config, preprocessing_config, dimensionality_config
from .pipelines import get_pipelines
from .utils import score_candidate

from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted


class AutoClustering:
    """
    Automatic Clustering Selection with Ray Tune

    Parameters
    ----------
    num_samples : int, default=100
        Number of times to sample from the hyperparameter space.
    metric : str, default='validity_index'
        Metric to optimize.
    verbose : int, default=0
        Verbosity mode.
        0 = silent,
        1 = only status updates,
        2 = status and brief results,
        3 = status and detailed results.
    n_jobs: int, default=1 Maximum number of trials to run
            concurrently. Must be non-negative. If None or 0, no limit will
            be applied.

    Returns
    -------
    score: dict
        Dict with popular clustering metrics
    """

    def __init__(self,
                 num_samples: int = 100,
                 metric: str = 'validity_index',
                 verbose=0,
                 n_jobs=1):
        self.num_samples = num_samples
        self.metric = metric

        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X):
        """
        Main method of AutoClustering, it runs the optimization with ray and optuna

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        """
        search_space = get_pipelines(preprocessing_config, clustering_config, dimensionality_config)

        config = {"algorithm": tune.choice(list(search_space.keys())),
                  "search_space": search_space}

        tuner = tune.Tuner(tune.with_parameters(self._train, X=X),
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

    def fit_predict(self, X):
        """
        Compute cluster and predict cluster index for each sample.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------

        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        return self.best_estimator_.fit_predict(X)

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------

        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        if isinstance(self.best_estimator_.named_steps["clustering"], HDBSCAN):
            temp_params = {**self.best_params_, "clustering__prediction_data": True}
            estimator = clone(self.best_estimator_)
            estimator.set_params(**temp_params)
            estimator.fit(X)
            return estimator.named_steps["clustering"].labels_

        return self.best_estimator_.predict(X)

    @staticmethod
    def _train(config, X):
        """
        Runs one iteration of the pipeline search

        Parameters
        ----------
        config : dict
            configuration sampled from the search space
        X : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        """
        algorithm = config["algorithm"]
        pipeline_dict = config["search_space"][algorithm]

        pipe = pipeline_dict["pipe"]
        params = pipeline_dict["params"]
        pipe.set_params(**params)
        pipe.fit(X)

        scores = score_candidate(pipe, X)
        tune.report(**scores)

    @property
    def optimization_mode_(self):
        """
        Returns
        -------
        Indicates whether to maximize or minimize the metric
        """
        modes_map = {"validity_index": 'max',
                     "davies_bouldin": 'min',
                     "silhouette": 'max',
                     'calinski_harabasz': 'max'}
        metric_mode = modes_map.get(self.metric)
        if not metric_mode:
            raise ValueError(f"Metric must be one of "
                             f"['validity_index', 'davies_bouldin', 'silhouette', 'calinski_harabasz'], "
                             f"but got {self.metric} instead")

        return metric_mode

    @property
    def n_clusters_(self):
        """
        Returns
        -------
        Number of clusters found during fit time
        """
        labels = self.best_estimator_.named_steps["clustering"].labels_

        return len(np.unique(labels))
