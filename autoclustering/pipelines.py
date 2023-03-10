from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid


def get_pipelines(preprocessing_config, dimensionality_config, cluster_config):
    pipelines = {}
    param_grid = {'preprocessing': preprocessing_config,
                  'dimensionality': dimensionality_config,
                  'clustering': cluster_config}
    param_list = list(ParameterGrid(param_grid))

    for i, param in enumerate(param_list):
        pipe = Pipeline([
            ("preprocessing", param["preprocessing"]["model"]),
            ("dimensionality", param["dimensionality"]["model"]),
            ("clustering", param["clustering"]["model"])
        ])

        preprocessing_params = param["preprocessing"].get("params")
        if preprocessing_params:
            preprocessing_params = {f"preprocessing__{key}": value for key, value in preprocessing_params.items()}

        dimensionality_params = param["dimensionality"].get("params")
        if dimensionality_params:
            dimensionality_params = {f"dimensionality__{key}": value for key, value in dimensionality_params.items()}

        clustering_params = param["clustering"].get("params")
        if clustering_params:
            clustering_params = {f"clustering__{key}": value for key, value in clustering_params.items()}

        param_space = {**preprocessing_params, **clustering_params, **dimensionality_params}

        pipelines[f"pipeline_{i}"] = {"pipe": pipe, "params": param_space}

    return pipelines
