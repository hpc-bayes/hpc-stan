from dask.distributed import Client, LocalCluster

from hpc_stan.hpc_bridgestan import HPCBridgeStanBase

cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=1)
client = Client(cluster)

bridge = HPCBridgeStanBase(
    "tests/models/bernoulli.stan",
    "tests/models/bernoulli.json",
    client=client,
)

try:
    log_density = client.gather(
        bridge.log_density([0.0], propto=False, jacobian=True)
    )
    gradient = client.gather(bridge.log_density_gradient([0.0]))

    print(log_density)
    print(gradient)
finally:
    client.close()
    cluster.close()
