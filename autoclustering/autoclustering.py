import warnings
import multiprocessing
import numpy as np
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from typing import List

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
    metric : str, {'validity_index', 'davies_bouldin', 'silhouette', 'calinski_harabasz'}, default='validity_index'
        Metric to optimize.
    preprocessing_models: dict, default=None
        Custom dict with the preprocessing models to create the pipelines
        If None, the default models will be used.
        Check autoclustering.configs.preprocessing as format example
    dimensionality_models: dict, default=None
        Custom dict with the dimensionality reduction models to create the pipelines
        If None, the default models will be used.
        Check autoclustering.configs.dimensionality as format example
    clustering_models: dict, default=None
        Custom dict with the clustering models to create the pipelines
        If None, the default models will be used.
        Check autoclustering.configs.clustering as format example
    verbose : int, default=0
        Verbosity mode.
        0 = silent,
        1 = only status updates,
        2 = status and brief results,
        3 = status and detailed results.
    n_jobs: int, default=1 Maximum number of trials to run
            concurrently. Must be non-negative. If None or 0, no limit will
            be applied.
    """

    def __init__(self,
                 num_samples: int = 100,
                 metric: str = 'validity_index',
                 preprocessing_models: List[dict] = None,
                 dimensionality_models: List[dict] = None,
                 clustering_models: List[dict] = None,
                 verbose=0,
                 n_jobs=1):
        self.num_samples = num_samples
        self.metric = metric
        self.preprocessing_models = preprocessing_models or preprocessing_config
        self.dimensionality_models = dimensionality_models or dimensionality_config
        self.clustering_models = clustering_models or clustering_config
        self.verbose = verbose
        self.n_jobs = int(n_jobs or -1)

        if self.n_jobs < 0:
            resources_per_trial = {"cpu": 1}
            if self.n_jobs < -1:
                warnings.warn(
                    "`self.n_jobs` is automatically set "
                    "-1 for any negative values.",
                    category=UserWarning)
        else:
            available_cpus = multiprocessing.cpu_count()
            if ray.is_initialized():  # pragma: no cover
                available_cpus = ray.cluster_resources()["CPU"]

            cpu_fraction = available_cpus / self.n_jobs
            if cpu_fraction > 1:
                cpu_fraction = int(np.ceil(cpu_fraction))

            resources_per_trial = {"cpu": cpu_fraction}

        self.resources_per_trial = resources_per_trial

    def fit(self, X):
        """
        Main method of AutoClustering, it runs the optimization with ray and optuna

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        """
        search_space = get_pipelines(self.preprocessing_models,
                                     self.dimensionality_models,
                                     self.clustering_models)

        config = {"algorithm": tune.choice(list(search_space.keys())),
                  "search_space": search_space}

        analysis = tune.run(tune.with_parameters(self._train, X=X),
                            config=config,
                            metric=self.metric,
                            mode=self.optimization_mode_,
                            num_samples=self.num_samples,
                            search_alg=OptunaSearch(),
                            resources_per_trial=self.resources_per_trial,
                            verbose=self.verbose)

        best_result = analysis.best_result

        # Refit best pipeline
        best_pipeline = best_result["config"]["algorithm"]
        best_model = best_result["config"]["search_space"][best_pipeline]

        self.best_estimator_ = best_model["pipe"]
        self.best_params_ = best_model["params"]
        self.best_estimator_.set_params(**self.best_params_).fit(X)
        self.best_score_ = best_result[self.metric]

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
        check_is_fitted(self.best_estimator_)

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
        check_is_fitted(self.best_estimator_)

        if isinstance(self.best_estimator_.named_steps["clustering"], HDBSCAN):
            temp_params = {**self.best_params_, "clustering__prediction_data": True}
            estimator = clone(self.best_estimator_)
            estimator.set_params(**temp_params)
            estimator.fit(X)
            return estimator.named_steps["clustering"].labels_
        elif isinstance(self.best_estimator_.named_steps["clustering"], DBSCAN):
            return self.fit_predict(X)

        return self.best_estimator_.predict(X)

    @staticmethod
    def _train(config, X):  # pragma: no cover
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
                             f"but got {self.metric} instead.")

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
