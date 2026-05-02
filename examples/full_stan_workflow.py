"""A complete local workflow using both BridgeStan and CmdStanPy wrappers.

Use this as the shape of a real analysis:
1. BridgeStan does fast in-memory checks of the target density.
2. CmdStanPy runs inference through the same Dask execution layer.
3. Generated quantities are run from the fitted CmdStanPy sample.

For HPC, replace BaseClusterCmdStanPy/HPCBridgeStanBase with matching
dask-jobqueue wrappers, such as CmdStanPySLURMCluster and SLURMClusterBridgeStan.
"""

import sys
from pathlib import Path

from dask.distributed import Client, LocalCluster

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_stan.hpc_bridgestan import HPCBridgeStanBase
from hpc_stan.hpc_cmdstanpy import BaseClusterCmdStanPy

BERNOULLI = "tests/models/bernoulli.stan"
BERNOULLI_GQ = "tests/models/bernoulli_gq.stan"
BERNOULLI_DATA = "tests/models/bernoulli.json"


cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=1)
client = Client(cluster)

try:
    bridge = HPCBridgeStanBase(BERNOULLI, BERNOULLI_DATA, client=client)

    density = client.gather(bridge.log_density([0.0], propto=False, jacobian=True))
    _, gradient = client.gather(
        bridge.log_density_gradient([0.0], propto=False, jacobian=True)
    )
    names = client.gather(bridge.param_names())

    print(f"BridgeStan parameters: {names}")
    print(f"BridgeStan log density at theta_unc=0: {density}")
    print(f"BridgeStan gradient at theta_unc=0: {gradient}")

    cmdstan = BaseClusterCmdStanPy(BERNOULLI, client=client)
    fits = cmdstan.sample(
        n_tasks=1,
        data=BERNOULLI_DATA,
        chains=2,
        iter_warmup=100,
        iter_sampling=100,
        seed=12345,
        show_progress=False,
    )

    mle = client.gather(cmdstan.optimize(data=BERNOULLI_DATA, algorithm="LBFGS"))
    vb = client.gather(
        cmdstan.variational(
            data=BERNOULLI_DATA,
            algorithm="meanfield",
            require_converged=False,
        )
    )

    gq_model = BaseClusterCmdStanPy(BERNOULLI_GQ, client=client)
    generated = client.gather(
        gq_model.generate_quantities(data=BERNOULLI_DATA, mcmc_sample=fits[0])
    )

    print(f"CmdStanPy sample variables: {list(fits[0].stan_variables())}")
    print(f"CmdStanPy optimized theta: {mle.optimized_params_dict['theta']}")
    print(f"CmdStanPy variational columns: {vb.column_names}")
    print(f"Generated quantities variables: {list(generated.stan_variables())}")
finally:
    client.close()
    cluster.close()
