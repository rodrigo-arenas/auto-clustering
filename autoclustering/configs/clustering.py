from sklearn.cluster import DBSCAN, AffinityPropagation, OPTICS
from ray import tune

clustering_config = [
    {"model": DBSCAN(), "params": {"eps": tune.loguniform(0.01, 0.9),
                                   "min_samples": tune.randint(5, 20)}
     },
    {"model": AffinityPropagation(), "params": {"damping": tune.uniform(0.5, 0.99),
                                                "convergence_iter": tune.randint(5, 20),
                                                "max_iter": tune.randint(150, 300)}
     },
    {"model": OPTICS(), "params": {"eps": tune.loguniform(0.01, 0.9),
                                   "cluster_method": tune.choice(["xi", "dbscan"]),
                                   "min_samples": tune.randint(5, 20),
                                   "xi": tune.loguniform(0.02, 0.9)}
     },
]
