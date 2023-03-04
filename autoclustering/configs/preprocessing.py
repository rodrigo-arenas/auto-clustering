from dirty_cat import TableVectorizer

preprocessing_config = [
    {"model": TableVectorizer(), "params": {}},
    {"model": 'passthrough', "params": {}}
]
