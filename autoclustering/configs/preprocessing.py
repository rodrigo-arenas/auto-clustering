from dirty_cat import TableVectorizer
from sklearn.preprocessing import RobustScaler, MinMaxScaler

preprocessing_config = [
    {"model": 'passthrough', "params": {}},
    {"model": TableVectorizer(), "params": {}},
    {"model": RobustScaler(), "params": {}},
    {"model": MinMaxScaler(), "params": {}}
]
