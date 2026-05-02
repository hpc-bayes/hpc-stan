from hpc_stan import SLURMClusterBridgeStan

cluster_kwargs = {
    "cores": 2,
    "memory": "2GB",
    "processes": 1,
    "queue": "regular",
    "walltime": "02:00:00",
    "interface": "ib0",
    "local_directory": "/scratch",
    "job_extra": ["-M myemail@my.domain", "-m abe"],
    "env_extra": [
        'export LANG="en_US.utf8"',
        'export LC_ALL="en_US.utf8"',
    ],
}

bridge = SLURMClusterBridgeStan(
    "/path/to/your/model.stan",
    "/path/to/your/data.json",
    cluster_kwargs=cluster_kwargs,
    n_jobs=2,
)

# BridgeStan is an in-memory model interface. This submits model-method calls
# to Dask workers on the already configured SLURM cluster.
log_density_future = bridge.log_density([0.0], propto=False, jacobian=True)
gradient_future = bridge.log_density_gradient([0.0])

log_density = bridge.get_client().gather(log_density_future)
gradient = bridge.get_client().gather(gradient_future)

print(log_density)
print(gradient)
