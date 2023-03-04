
auto-clustering
===============

Automatic Clustering selection with Ray Tune

**Important Note:**

This tool may optimize metrics such as the validity index, silhouette or the davies-bouldin score,
but as this is based on unsupervised learning, these metrics may not always reflect the true usefulness
of the resulting clusters.

Installation:
#############

Install autoclustering

It's advised to install auto-clustering using a virtual env, to install a light version,
inside the env use::

   pip install auto-clustering

.. |PythonMinVersion| replace:: 3.9
.. |ScikitLearnMinVersion| replace:: 1.1.0
.. |NumPyMinVersion| replace:: 1.22.0
.. |RayTuneMinVersion| replace:: 2.3.0
.. |OptunaMinVersion| replace:: 3.0.0
.. |DirtyCatMinVersion| replace:: 0.4.0
.. |HdbscanMinVersion| replace:: 0.8.27


sklearn-genetic-opt requires:

- Python (>= |PythonMinVersion|)
- scikit-learn (>= |ScikitLearnMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- Ray-Tune (>= |RayTuneMinVersion|)
- Optuna (>= |OptunaMinVersion|)
- HDBSCAN (>= |DirtyCatMinVersion|)
- Dirty-Cat (>= |HdbscanMinVersion|)


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: User Guide / Tutorials:

   tutorials/basic_usage

.. toctree::
   :maxdepth: 2
   :caption: Release Notes

   release_notes

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/autoclustering


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
