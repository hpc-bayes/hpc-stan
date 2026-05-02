from hpc_stan import MOABClusterBridgeStan

cluster_kwargs = {
    "cores": 2,
    "memory": "2GB",
    "processes": 1,
    "queue": "regular",
    "resource_spec": "select=1:ncpus=2:mem=2GB",
    "project": "my_project",
    "walltime": "02:00:00",
    "interface": "ib0",
    "local_directory": "/scratch",
    "job_extra": ["-M myemail@my.domain", "-m abe"],
    "env_extra": [
        'export LANG="en_US.utf8"',
        'export LC_ALL="en_US.utf8"',
    ],
}

bridge = MOABClusterBridgeStan(
    "/path/to/your/model.stan",
    "/path/to/your/data.json",
    cluster_kwargs=cluster_kwargs,
    n_jobs=2,
)

param_names_future = bridge.param_names()
log_density_future = bridge.log_density([0.0], propto=False, jacobian=True)

param_names = bridge.get_client().gather(param_names_future)
log_density = bridge.get_client().gather(log_density_future)

print(param_names)
print(log_density)
