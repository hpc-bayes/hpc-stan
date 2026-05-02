"""CmdStanPy methods through hpc-stan's Dask wrapper.

This example assumes:
1. CmdStan is installed.
2. The Stan files under tests/models are available.
3. You are using a local Dask cluster for demonstration.

On an HPC system, replace BaseClusterCmdStanPy with a jobqueue wrapper such as
CmdStanPySLURMCluster and pass scheduler settings through cluster_kwargs.
"""

import sys
from pathlib import Path

from dask.distributed import Client, LocalCluster

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_stan.hpc_cmdstanpy import BaseClusterCmdStanPy

BERNOULLI = "tests/models/bernoulli.stan"
BERNOULLI_GQ = "tests/models/bernoulli_gq.stan"
BERNOULLI_DATA = "tests/models/bernoulli.json"
ROSENBROCK = "tests/models/rosenbrock.stan"


cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=1)
client = Client(cluster)

try:
    bernoulli = BaseClusterCmdStanPy(BERNOULLI, client=client)

    # Compilation and model metadata helpers.
    compile_future = bernoulli.compile()
    code_future = bernoulli.code()
    exe_info_future = bernoulli.exe_info()
    src_info_future = bernoulli.src_info()

    # MCMC sampling. sample() is the one convenience method that gathers a list
    # of results because it often launches repeated model fits.
    fits = bernoulli.sample(
        n_tasks=2,
        data=BERNOULLI_DATA,
        chains=1,
        iter_warmup=100,
        iter_sampling=100,
        seed=12345,
        show_progress=False,
    )

    # Diagnostics on a fitted model run.
    diagnose_future = bernoulli.diagnose(data=BERNOULLI_DATA)

    # Optimization.
    mle_future = bernoulli.optimize(
        data=BERNOULLI_DATA,
        algorithm="LBFGS",
        seed=12345,
    )

    # Variational inference.
    vb_future = bernoulli.variational(
        data=BERNOULLI_DATA,
        algorithm="meanfield",
        seed=12345,
        require_converged=False,
    )

    # Pathfinder.
    pathfinder_future = bernoulli.pathfinder(
        data=BERNOULLI_DATA,
        seed=12345,
    )

    # Generated quantities use a second Stan program and an existing MCMC fit.
    gq_model = BaseClusterCmdStanPy(BERNOULLI_GQ, client=client)
    gq_future = gq_model.generate_quantities(
        data=BERNOULLI_DATA,
        mcmc_sample=fits[0],
    )

    # Laplace sampling typically uses an optimized mode.
    laplace_future = gq_model.laplace_sample(
        data=BERNOULLI_DATA,
        mode=mle_future,
    )

    # log_prob is available through CmdStanPy. Its exact accepted arguments can
    # vary across CmdStanPy releases, so pass the same arguments you would pass
    # to CmdStanModel.log_prob.
    log_prob_future = bernoulli.submit(
        "log_prob",
        data=BERNOULLI_DATA,
        params={"theta": 0.5},
    )

    results = client.gather(
        [
            compile_future,
            code_future,
            exe_info_future,
            src_info_future,
            diagnose_future,
            mle_future,
            vb_future,
            pathfinder_future,
            gq_future,
            laplace_future,
            log_prob_future,
        ]
    )

    print(f"sample tasks: {len(fits)}")
    print(f"gathered method results: {len(results)}")
finally:
    client.close()
    cluster.close()
