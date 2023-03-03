import os
import pathlib
from setuptools import setup, find_packages

# python setup.py sdist bdist_wheel
# twine upload --skip-existing dist/*

# get __version__ from _version.py
ver_file = os.path.join("sklearn_genetic", "_version.py")
with open(ver_file) as f:
    exec(f.read())

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.rst").read_text()
setup(
    name="sklearn-genetic-opt",
    version=__version__,
    description="Automatic Clustering Selection",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/rodrigo-arenas/auto-clustering",
    author="Rodrigo Arenas",
    author_email="rodrigo.arenas456@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    project_urls={
        "Documentation": "https://auto-clustering.readthedocs.io/en/stable/",
        "Source Code": "https://github.com/rodrigo-arenas/auto-clustering",
        "Bug Tracker": "https://github.com/rodrigo-arenas/auto-clustering/issues",
    },
    packages=find_packages(
        include=["autoclustering", "autoclustering.*"], exclude=["*tests*"]
    ),
    install_requires=[
        "scikit-learn>=1.1.0",
        "numpy>=1.19.0",
        "ray[tune]>=2.3.0",
        "optuna>=3.0.0",
        "dirty-cat>=0.4.0",
    ],
    python_requires=">=3.9",
    include_package_data=True,
)