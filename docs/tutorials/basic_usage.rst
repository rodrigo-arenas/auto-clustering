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
