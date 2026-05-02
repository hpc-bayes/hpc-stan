from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import CMDSTANPY_AVAILABLE, DASK_DISTRIBUTED_AVAILABLE

MODELS_DIR = Path(__file__).parent / "models"
BERNOULLI_STAN = MODELS_DIR / "bernoulli.stan"
BERNOULLI_GQ_STAN = MODELS_DIR / "bernoulli_gq.stan"
BERNOULLI_JSON = MODELS_DIR / "bernoulli.json"
BERNOULLI_INIT = MODELS_DIR / "bernoulli.init.json"
ROSENBROCK_STAN = MODELS_DIR / "rosenbrock.stan"

SAMPLER_STATE = (
    "lp__",
    "accept_stat__",
    "stepsize__",
    "treedepth__",
    "n_leapfrog__",
    "divergent__",
    "energy__",
)
BERNOULLI_COLS = SAMPLER_STATE + ("theta",)


@pytest.fixture(scope="module")
def dask_client():
    if not DASK_DISTRIBUTED_AVAILABLE:
        pytest.skip("dask.distributed is not installed")
    from dask.distributed import Client

    client = Client(processes=False, n_workers=1, threads_per_worker=1)
    yield client
    client.close()


@pytest.fixture(scope="module")
def cmdstan_ready():
    if not CMDSTANPY_AVAILABLE:
        pytest.skip("cmdstanpy is not installed")
    import cmdstanpy

    try:
        cmdstanpy.cmdstan_path()
    except (ValueError, RuntimeError):
        pytest.skip("CmdStan is not installed; run python -m cmdstanpy.install_cmdstan")


def _gather(client, future):
    return client.gather(future)


@pytest.mark.stan
def test_sample_bernoulli_good_ported_from_cmdstanpy(cmdstan_ready, dask_client):
    from hpc_stan.hpc_cmdstanpy import BaseClusterCmdStanPy

    wrapper = BaseClusterCmdStanPy(str(BERNOULLI_STAN), client=dask_client)
    fits = wrapper.sample(
        n_tasks=1,
        data=str(BERNOULLI_JSON),
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=20,
        iter_sampling=20,
        show_progress=False,
    )
    fit = fits[0]

    assert "method=sample" in repr(fit)
    assert fit.chains == 2
    assert fit.thin == 1
    assert fit.num_draws_warmup == 20
    assert fit.num_draws_sampling == 20
    assert fit.column_names == BERNOULLI_COLS
    assert fit.draws().shape == (20, 2, len(BERNOULLI_COLS))
    assert fit.draws(concat_chains=True).shape == (40, len(BERNOULLI_COLS))
    assert "theta" in fit.stan_variables()
    assert fit.step_size is not None


@pytest.mark.stan
def test_sample_accepts_dict_data_and_inits_like_cmdstanpy(cmdstan_ready, dask_client):
    from hpc_stan.hpc_cmdstanpy import BaseClusterCmdStanPy

    wrapper = BaseClusterCmdStanPy(str(BERNOULLI_STAN), client=dask_client)
    fits = wrapper.sample(
        n_tasks=1,
        data={"N": 10, "y": [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]},
        inits={"theta": 0.2},
        chains=1,
        seed=12345,
        iter_warmup=10,
        iter_sampling=10,
        show_progress=False,
    )

    assert fits[0].draws().shape == (10, 1, len(BERNOULLI_COLS))


@pytest.mark.stan
def test_optimize_bernoulli_good_ported_from_cmdstanpy(cmdstan_ready, dask_client):
    from hpc_stan.hpc_cmdstanpy import BaseClusterCmdStanPy

    wrapper = BaseClusterCmdStanPy(str(BERNOULLI_STAN), client=dask_client)
    mle = _gather(
        dask_client,
        wrapper.optimize(
            data=str(BERNOULLI_JSON),
            inits=str(BERNOULLI_INIT),
            seed=1239812093,
            algorithm="LBFGS",
            iter=100,
        ),
    )

    assert "method=optimize" in repr(mle)
    assert "theta" in mle.metadata.stan_vars
    assert mle.stan_variable("theta").shape == ()
    np.testing.assert_almost_equal(mle.optimized_params_dict["theta"], 0.5, decimal=2)


@pytest.mark.stan
def test_optimize_rosenbrock_ported_from_cmdstanpy(cmdstan_ready, dask_client):
    from hpc_stan.hpc_cmdstanpy import BaseClusterCmdStanPy

    wrapper = BaseClusterCmdStanPy(str(ROSENBROCK_STAN), client=dask_client)
    mle = _gather(dask_client, wrapper.optimize(seed=1239812093, algorithm="BFGS"))

    assert mle.column_names == ("lp__", "x", "y")
    np.testing.assert_almost_equal(mle.optimized_params_dict["x"], 1, decimal=3)
    np.testing.assert_almost_equal(mle.optimized_params_dict["y"], 1, decimal=3)


@pytest.mark.stan
def test_variational_bernoulli_accessors_ported_from_cmdstanpy(cmdstan_ready, dask_client):
    from hpc_stan.hpc_cmdstanpy import BaseClusterCmdStanPy

    wrapper = BaseClusterCmdStanPy(str(BERNOULLI_STAN), client=dask_client)
    vb = _gather(
        dask_client,
        wrapper.variational(
            data=str(BERNOULLI_JSON),
            seed=12345,
            algorithm="meanfield",
            require_converged=False,
        ),
    )

    assert "method=variational" in repr(vb)
    assert "theta" in vb.metadata.stan_vars
    assert vb.stan_variable("theta", mean=False).ndim == 1
    assert "theta" in vb.variational_params_dict


@pytest.mark.stan
def test_generate_quantities_ported_from_cmdstanpy(cmdstan_ready, dask_client):
    from hpc_stan.hpc_cmdstanpy import BaseClusterCmdStanPy

    sample_wrapper = BaseClusterCmdStanPy(str(BERNOULLI_STAN), client=dask_client)
    fit = sample_wrapper.sample(
        n_tasks=1,
        data=str(BERNOULLI_JSON),
        chains=1,
        seed=12345,
        iter_warmup=10,
        iter_sampling=10,
        show_progress=False,
    )[0]

    gq_wrapper = BaseClusterCmdStanPy(str(BERNOULLI_GQ_STAN), client=dask_client)
    gq = _gather(
        dask_client,
        gq_wrapper.generate_quantities(data=str(BERNOULLI_JSON), mcmc_sample=fit),
    )

    assert gq.draws().shape[0] == 10
    assert "y_rep" in gq.stan_variables()
