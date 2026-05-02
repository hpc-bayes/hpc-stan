"""Template showing every BridgeStan method on every jobqueue cluster wrapper.

Replace cluster_kwargs with values from your scheduler. BridgeStan is an
in-memory interface, so these methods evaluate model metadata, transforms, log
density, gradients, and Hessian-related calls through Dask workers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from hpc_stan import (
        HTCondorClusterBridgeStan,
        LSFClusterBridgeStan,
        MOABClusterBridgeStan,
        OARClusterBridgeStan,
        PBSClusterBridgeStan,
        SGEClusterBridgeStan,
        SLURMClusterBridgeStan,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Install hpc-stan requirements before running this template: "
        "python -m pip install -r requirements.txt"
    ) from exc

STAN_FILE = "tests/models/bernoulli.stan"
DATA_FILE = "tests/models/bernoulli.json"
THETA_UNC = [0.0]


CLUSTERS = {
    "SLURM": (SLURMClusterBridgeStan, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "PBS": (PBSClusterBridgeStan, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "SGE": (SGEClusterBridgeStan, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "LSF": (LSFClusterBridgeStan, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "OAR": (OARClusterBridgeStan, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "Moab": (MOABClusterBridgeStan, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "HTCondor": (HTCondorClusterBridgeStan, {"cores": 2, "memory": "4GB"}),
}


def submit_all_bridgestan_methods(cluster_cls, cluster_kwargs):
    model = cluster_cls(
        STAN_FILE,
        DATA_FILE,
        cluster_kwargs=cluster_kwargs,
        n_jobs=2,
    )

    futures = [
        model.name(),
        model.model_info(),
        model.param_num(),
        model.param_unc_num(),
        model.param_names(),
        model.param_unc_names(),
        model.param_constrain(THETA_UNC),
        model.param_unconstrain([0.5]),
        model.param_unconstrain_json({"theta": 0.5}),
        model.log_density(THETA_UNC, propto=False, jacobian=True),
        model.log_density_gradient(THETA_UNC, propto=False, jacobian=True),
        model.log_density_hessian(THETA_UNC, propto=False, jacobian=True),
        model.log_density_hessian_vector_product(THETA_UNC, [1.0]),
        model.new_rng(seed=1234),
    ]

    return model.get_client().gather(futures)


if __name__ == "__main__":
    for scheduler, (cluster_cls, _) in CLUSTERS.items():
        print(f"{scheduler}: {cluster_cls.__name__}")
        print("  methods: name, model_info, param_num, param_unc_num,")
        print("           param_names, param_unc_names, param_constrain,")
        print("           param_unconstrain, param_unconstrain_json,")
        print("           log_density, log_density_gradient,")
        print("           log_density_hessian,")
        print("           log_density_hessian_vector_product, new_rng")
