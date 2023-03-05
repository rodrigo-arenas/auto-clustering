How to Use auto-clustering
==========================

auto-clustering performs metric optimization using ray-tune
By defaults it search among a set of pipelines which contains common preprocessing,
dimensionality reduction and clustering techniques.

Make sure to check the parameters that you can set in the API docs.

Example: Clustering Selection
###############################

.. code-block:: python

   from autoclustering import AutoClustering
   from sklearn.datasets import load_digits


   data, _ = load_digits(return_X_y=True)

   clustering = AutoClustering(num_samples=50,
                               metric='validity_index',
                               n_jobs=2,
                               verbose=0)

   clustering.fit(data)

   clustering.best_params_
   clustering.best_score_
   clustering.n_clusters_
   clustering.best_estimator_

   clustering.predict(data)

You can also define your own preprocessing, dimensionality reduction and clustering steps.
This can be done using the parameters preprocessing_models, dimensionality_models and clustering_models.
The models must be compatible with the scikit-learn API.
The following is an example defining a custom DBSCAN and HDBSCAN search models.

Example: Custom Model
#####################

.. code-block:: python

   from autoclustering import AutoClustering
   from ray import tune
   from sklearn.datasets import load_digits
   from sklearn.cluster import DBSCAN
   from hdbscan import HDBSCAN

   models_config = [
        {"model": DBSCAN(), "params": {"eps": tune.loguniform(0.5, 4),
                                        "min_samples": tune.randint(5, 20)}},
        {"model": HDBSCAN(), "params": {"min_cluster_size": tune.randint(5, 30),
                                         "cluster_selection_method": tune.choice(["eom", "leaf"])}
        }]


   data, _ = load_digits(return_X_y=True)

   clustering = AutoClustering(num_samples=50,
                               metric='validity_index',
                               clustering_models=models_config,
                               n_jobs=2,
                               verbose=0)

   clustering.fit(data)

   clustering.best_params_
   clustering.best_score_
   clustering.n_clusters_
   clustering.best_estimator_

   clustering.predict(data)