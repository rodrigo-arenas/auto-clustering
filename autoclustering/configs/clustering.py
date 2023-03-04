from ray import tune
from sklearn.cluster import DBSCAN, AffinityPropagation, OPTICS
from hdbscan import HDBSCAN

clustering_config = [
    {"model": DBSCAN(), "params": {"eps": tune.loguniform(0.5, 4),
                                   "min_samples": tune.randint(5, 20)}
     },
    {"model": AffinityPropagation(), "params": {"damping": tune.uniform(0.5, 0.99),
                                                "convergence_iter": tune.randint(5, 20),
                                                "max_iter": tune.randint(150, 300)}
     },
    {"model": OPTICS(), "params": {"eps": tune.loguniform(0.5, 4),
                                   "cluster_method": tune.choice(["xi", "dbscan"]),
                                   "min_samples": tune.randint(5, 20),
                                   "xi": tune.loguniform(0.02, 0.9)}
     },
    {"model": HDBSCAN(), "params": {"min_cluster_size": tune.randint(5, 30),
                                    "cluster_selection_method": tune.choice(["eom", "leaf"])}
     }

]
