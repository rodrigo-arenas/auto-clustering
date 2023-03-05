.. -*- mode: rst -*-

|Tests|_ |Codecov|_ |PythonVersion|_ |PyPi|_ |Docs|_

.. |Tests| image:: https://github.com/rodrigo-arenas/auto-clustering/actions/workflows/ci-tests.yml/badge.svg?branch=main
.. _Tests: https://github.com/rodrigo-arenas/auto-clustering/actions/workflows/ci-tests.yml

.. |Codecov| image:: https://codecov.io/gh/rodrigo-arenas/auto-clustering/branch/master/graphs/badge.svg?branch=master&service=github
.. _Codecov: https://codecov.io/github/rodrigo-arenas/auto-clustering?branch=main

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.9%20%7C%203.10-blue
.. _PythonVersion : https://www.python.org/downloads/
.. |PyPi| image:: https://badge.fury.io/py/auto-clustering.svg
.. _PyPi: https://badge.fury.io/py/auto-clustering

.. |Docs| image:: https://readthedocs.org/projects/auto-clustering/badge/?version=latest
.. _Docs: https://auto-clustering.readthedocs.io/en/latest/?badge=latest

.. |Contributors| image:: https://contributors-img.web.app/image?repo=rodrigo-arenas/auto-clustering
.. _Contributors: https://github.com/rodrigo-arenas/auto-clustering/graphs/contributors

auto-clustering
###############

Automatic Clustering selection with Ray Tune

**Important Note:**

This tool may optimize metrics such as the validity index, silhouette or the davies-bouldin score,
but as this is based on unsupervised learning, these metrics may not always reflect the true usefulness
of the resulting clusters.

Example: Clustering Selection
###############################

.. code-block:: python

   from autoclustering import AutoClustering
   from sklearn.datasets import load_digits


   data, _ = load_digits(return_X_y=True)

   clustering = AutoClustering(num_samples=50,
                               metric='validity_index',
                               n_jobs=-1,
                               verbose=0)

   clustering.fit(data)

   clustering.best_params_
   clustering.best_score_
   clustering.n_clusters_
   clustering.best_estimator_

   clustering.predict(data)

Changelog
#########

See the `changelog <https://auto-clustering.readthedocs.io/en/latest/release_notes.html>`__
for notes on the changes of auto-clustering

Important links
###############

- Official source code repo: https://github.com/rodrigo-arenas/auto-clustering/
- Download releases: https://pypi.org/project/auto-clustering/
- Issue tracker: https://github.com/rodrigo-arenas/auto-clustering/issues
- Stable documentation: https://auto-clustering.readthedocs.io/en/stable/

Source code
###########

You can check the latest development version with the command::

   git clone https://github.com/rodrigo-arenas/auto-clustering.git

Install the development dependencies::

  pip install -r requirements.txt

Check the latest in-development documentation: https://auto-clustering.readthedocs.io/en/latest/

Contributing
############

Contributions are more than welcome!

There are several opportunities on the ongoing project, so please get in touch if you would like to help out.
Make sure to check the current issues and also
the `Contribution guide <https://github.com/rodrigo-arenas/auto-clustering/blob/master/CONTRIBUTING.md>`_.

Big thanks to the people who are helping with this project!

|Contributors|_

Testing
#######

After installation, you can launch the test suite from outside the source directory::

   pytest autoclustering/



Disclaimer
##########

The library is still experimental and under heavy development