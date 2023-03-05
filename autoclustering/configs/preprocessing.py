from dirty_cat import TableVectorizer
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from ray import tune

preprocessing_config = [
    {"model": 'passthrough', "params": {}},
    {"model": TableVectorizer(), "params": {}},
    {"model": RobustScaler(), "params": {"with_centering": tune.choice([True, False]),
                                         "unit_variance": tune.choice([True, False])}},
    {"model": MinMaxScaler(), "params": {}}
]
