from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import BRIDGESTAN_AVAILABLE, DASK_DISTRIBUTED_AVAILABLE

MODELS_DIR = Path(__file__).parent / "models"
BERNOULLI_STAN = MODELS_DIR / "bernoulli.stan"
BERNOULLI_JSON = MODELS_DIR / "bernoulli.json"
STDNORMAL_STAN = MODELS_DIR / "stdnormal.stan"


@pytest.fixture(scope="module")
def dask_client():
    if not DASK_DISTRIBUTED_AVAILABLE:
        pytest.skip("dask.distributed is not installed")
    from dask.distributed import Client

    client = Client(processes=False, n_workers=1, threads_per_worker=1)
    yield client
    client.close()


@pytest.fixture(scope="module")
def bridgestan_ready():
    if not BRIDGESTAN_AVAILABLE:
        pytest.skip("bridgestan is not installed")


def _gather(client, future):
    return client.gather(future)


def _logit(p):
    return np.log(p / (1 - p))


def _bernoulli_log_density(y, p):
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def _bernoulli_log_jacobian(p):
    return np.log(p * (1 - p))


@pytest.mark.stan
def test_constructor_and_param_metadata_ported_from_bridgestan(bridgestan_ready, dask_client):
    from hpc_stan.hpc_bridgestan import HPCBridgeStanBase

    wrapper = HPCBridgeStanBase(str(BERNOULLI_STAN), str(BERNOULLI_JSON), client=dask_client)
    model = wrapper.compile_model()

    assert model is not None
    assert _gather(dask_client, wrapper.param_unc_num()) == 1
    assert _gather(dask_client, wrapper.param_num(include_tp=False, include_gq=False)) == 1
    np.testing.assert_array_equal(_gather(dask_client, wrapper.param_names()), ["theta"])
    np.testing.assert_array_equal(_gather(dask_client, wrapper.param_unc_names()), ["theta"])


@pytest.mark.stan
def test_log_density_ported_from_bridgestan(bridgestan_ready, dask_client):
    from hpc_stan.hpc_bridgestan import HPCBridgeStanBase

    wrapper = HPCBridgeStanBase(str(BERNOULLI_STAN), str(BERNOULLI_JSON), client=dask_client)
    y = np.asarray([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
    p = np.asarray([0.3])
    q = _logit(p)

    lp = _gather(dask_client, wrapper.log_density(q, propto=False, jacobian=False))
    lp_jac = _gather(dask_client, wrapper.log_density(q, propto=False, jacobian=True))

    np.testing.assert_allclose(lp, _bernoulli_log_density(y, p))
    np.testing.assert_allclose(lp_jac, _bernoulli_log_density(y, p) + _bernoulli_log_jacobian(p))


@pytest.mark.stan
def test_log_density_gradient_ported_from_bridgestan(bridgestan_ready, dask_client):
    from hpc_stan.hpc_bridgestan import HPCBridgeStanBase

    wrapper = HPCBridgeStanBase(str(BERNOULLI_STAN), str(BERNOULLI_JSON), client=dask_client)
    p = np.asarray([0.3])
    q = _logit(p)

    lp, grad = _gather(dask_client, wrapper.log_density_gradient(q, propto=False, jacobian=False))

    assert np.isfinite(lp)
    assert grad.shape == (1,)


@pytest.mark.stan
def test_param_constrain_round_trip_ported_from_bridgestan(bridgestan_ready, dask_client):
    from hpc_stan.hpc_bridgestan import HPCBridgeStanBase

    wrapper = HPCBridgeStanBase(str(BERNOULLI_STAN), str(BERNOULLI_JSON), client=dask_client)
    p = np.asarray([0.7])
    q = _logit(p)

    constrained = _gather(dask_client, wrapper.param_constrain(q, include_tp=False, include_gq=False))
    unconstrained = _gather(dask_client, wrapper.param_unconstrain(constrained))

    np.testing.assert_allclose(constrained, p)
    np.testing.assert_allclose(unconstrained, q)


@pytest.mark.stan
def test_stdnormal_log_density_gradient_ported_from_bridgestan(bridgestan_ready, dask_client):
    from hpc_stan.hpc_bridgestan import HPCBridgeStanBase

    wrapper = HPCBridgeStanBase(str(STDNORMAL_STAN), client=dask_client)
    q = np.asarray([0.25])

    lp, grad = _gather(dask_client, wrapper.log_density_gradient(q, propto=True, jacobian=True))

    np.testing.assert_allclose(lp, -0.5 * q[0] ** 2)
    np.testing.assert_allclose(grad, -q)
