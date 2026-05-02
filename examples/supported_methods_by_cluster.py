"""Show supported hpc-stan clusters and forwarded backend methods.

This script is intentionally introspective: it documents the method surface
that hpc-stan forwards through Dask for every supported dask-jobqueue cluster
type. It does not create scheduler jobs.
"""

CMDSTANPY_MODEL_METHODS = (
    "code",
    "compile",
    "diagnose",
    "exe_info",
    "format",
    "generate_quantities",
    "laplace_sample",
    "log_prob",
    "optimize",
    "pathfinder",
    "sample",
    "src_info",
    "variational",
)

BRIDGESTAN_MODEL_METHODS = (
    "log_density",
    "log_density_gradient",
    "log_density_hessian",
    "log_density_hessian_vector_product",
    "model_info",
    "name",
    "new_rng",
    "param_constrain",
    "param_names",
    "param_num",
    "param_unc_names",
    "param_unc_num",
    "param_unconstrain",
    "param_unconstrain_json",
)


CMDSTANPY_CLUSTERS = {
    "SLURM": "CmdStanPySLURMCluster",
    "PBS": "CmdStanPyPBSCluster",
    "SGE": "CmdStanPySGECluster",
    "LSF": "CmdStanPyLSFCluster",
    "OAR": "CmdStanPyOARCluster",
    "Moab": "CmdStanPyMOABCluster",
    "HTCondor": "CmdStanPyHTCondorCluster",
}

BRIDGESTAN_CLUSTERS = {
    "SLURM": "SLURMClusterBridgeStan",
    "PBS": "PBSClusterBridgeStan",
    "SGE": "SGEClusterBridgeStan",
    "LSF": "LSFClusterBridgeStan",
    "OAR": "OARClusterBridgeStan",
    "Moab": "MOABClusterBridgeStan",
    "HTCondor": "HTCondorClusterBridgeStan",
}


def print_matrix(title, clusters, methods):
    print(title)
    print("=" * len(title))
    for scheduler, wrapper_name in clusters.items():
        print(f"\n{scheduler}: {wrapper_name}")
        for method in methods:
            print(f"  - {method}")


if __name__ == "__main__":
    print_matrix(
        "CmdStanPy Methods by Cluster",
        CMDSTANPY_CLUSTERS,
        CMDSTANPY_MODEL_METHODS,
    )
    print()
    print_matrix(
        "BridgeStan Methods by Cluster",
        BRIDGESTAN_CLUSTERS,
        BRIDGESTAN_MODEL_METHODS,
    )
