Release Notes
=============

Some notes on new features in various releases

What's new in 0.1.0dev0
-----------------------

^^^^^^^^^
Features:
^^^^^^^^^

* Introducing :class:`~autoclustering.AutoClustering` to select among a different
  set of clustering pipelines
* Pipelines selection using ray-tune, Optuna
* Optimize for metrics as 'validity_index', 'davies_bouldin', 'silhouette', 'calinski_harabasz'
* Pipelines composition from preprocessing, dimensionality reduction and clustering algorithms
* Enable custom preprocessing, dimensionality reduction and clustering algorithms pipelines