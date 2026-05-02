"""Template showing every CmdStanPy method on every jobqueue cluster wrapper.

Replace the scheduler-specific cluster_kwargs with values from your site. This
file demonstrates the hpc-stan API shape; it does not configure the scheduler
for you.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from hpc_stan import (
        CmdStanPyHTCondorCluster,
        CmdStanPyLSFCluster,
        CmdStanPyMOABCluster,
        CmdStanPyOARCluster,
        CmdStanPyPBSCluster,
        CmdStanPySGECluster,
        CmdStanPySLURMCluster,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Install hpc-stan requirements before running this template: "
        "python -m pip install -r requirements.txt"
    ) from exc

STAN_FILE = "tests/models/bernoulli.stan"
GQ_STAN_FILE = "tests/models/bernoulli_gq.stan"
DATA_FILE = "tests/models/bernoulli.json"


CLUSTERS = {
    "SLURM": (CmdStanPySLURMCluster, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "PBS": (CmdStanPyPBSCluster, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "SGE": (CmdStanPySGECluster, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "LSF": (CmdStanPyLSFCluster, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "OAR": (CmdStanPyOARCluster, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "Moab": (CmdStanPyMOABCluster, {"cores": 2, "memory": "4GB", "walltime": "01:00:00"}),
    "HTCondor": (CmdStanPyHTCondorCluster, {"cores": 2, "memory": "4GB"}),
}


def submit_all_cmdstanpy_methods(cluster_cls, cluster_kwargs):
    model = cluster_cls(STAN_FILE, cluster_kwargs=cluster_kwargs, n_jobs=2)

    # Model-management methods.
    compile_future = model.compile()
    code_future = model.code()
    exe_info_future = model.exe_info()
    src_info_future = model.src_info()

    # Inference methods.
    fits = model.sample(
        n_tasks=1,
        data=DATA_FILE,
        chains=1,
        iter_warmup=100,
        iter_sampling=100,
        show_progress=False,
    )
    optimize_future = model.optimize(data=DATA_FILE, algorithm="LBFGS")
    variational_future = model.variational(
        data=DATA_FILE,
        algorithm="meanfield",
        require_converged=False,
    )
    pathfinder_future = model.pathfinder(data=DATA_FILE)

    # Result-dependent methods.
    diagnose_future = model.diagnose(data=DATA_FILE)
    log_prob_future = model.submit("log_prob", data=DATA_FILE, params={"theta": 0.5})

    gq_model = cluster_cls(GQ_STAN_FILE, cluster_kwargs=cluster_kwargs, n_jobs=2)
    gq_future = gq_model.generate_quantities(data=DATA_FILE, mcmc_sample=fits[0])
    laplace_future = gq_model.laplace_sample(data=DATA_FILE, mode=optimize_future)

    return model.get_client().gather(
        [
            compile_future,
            code_future,
            exe_info_future,
            src_info_future,
            optimize_future,
            variational_future,
            pathfinder_future,
            diagnose_future,
            log_prob_future,
            gq_future,
            laplace_future,
        ]
    )


if __name__ == "__main__":
    for scheduler, (cluster_cls, cluster_kwargs) in CLUSTERS.items():
        print(f"{scheduler}: {cluster_cls.__name__}")
        print("  methods: compile, code, exe_info, src_info, sample, optimize,")
        print("           variational, pathfinder, diagnose, log_prob,")
        print("           generate_quantities, laplace_sample")
