from ray import tune
from sklearn.decomposition import IncrementalPCA, FastICA

dimensionality_config = [
    {"model": 'passthrough', "params": {}},
    {"model": IncrementalPCA(), "params": {"n_components": tune.randint(2, 10)}},
    {"model": FastICA(), "params": {"n_components": tune.randint(2, 10)}},

]
