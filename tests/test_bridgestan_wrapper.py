from __future__ import annotations

from hpc_stan.hpc_bridgestan import HPCBridgeStanBase, PBSClusterBridgeStan


class ImmediateClient:
    def submit(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def gather(self, futures):
        return list(futures)


class FakeStanModel:
    calls = []

    def __init__(self, stan_file, data, **kwargs):
        self.stan_file = stan_file
        self.data = data
        self.kwargs = kwargs

    @classmethod
    def from_stan_file(cls, stan_file, data=None, **kwargs):
        cls.calls.append(("from_stan_file", stan_file, data, kwargs))
        return cls(stan_file, data, **kwargs)

    def log_density(self, theta_unc, propto=True, jacobian=True):
        self.calls.append(("log_density", theta_unc, propto, jacobian))
        return -12.5

    def param_names(self, include_tp=False, include_gq=False):
        self.calls.append(("param_names", include_tp, include_gq))
        return ["theta"]


class FakeCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_compile_model_uses_bridgestan_from_stan_file(monkeypatch):
    import hpc_stan.hpc_bridgestan as module

    FakeStanModel.calls = []
    monkeypatch.setattr(module, "StanModel", FakeStanModel)

    wrapper = HPCBridgeStanBase(
        "bernoulli.stan",
        {"N": 2, "y": [0, 1]},
        seed=42,
        stanc_args=["--O1"],
        make_args=["STAN_THREADS=true"],
    )

    model = wrapper.compile_model()

    assert isinstance(model, FakeStanModel)
    assert FakeStanModel.calls == [
        (
            "from_stan_file",
            "bernoulli.stan",
            {"N": 2, "y": [0, 1]},
            {
                "seed": 42,
                "capture_stan_prints": True,
                "stanc_args": ["--O1"],
                "make_args": ["STAN_THREADS=true"],
            },
        )
    ]


def test_log_density_dispatches_bridgestan_method_through_dask(monkeypatch):
    import hpc_stan.hpc_bridgestan as module

    FakeStanModel.calls = []
    monkeypatch.setattr(module, "StanModel", FakeStanModel)

    wrapper = HPCBridgeStanBase("bernoulli.stan", client=ImmediateClient())

    result = wrapper.log_density([0.1], propto=False, jacobian=False)

    assert result == -12.5
    assert FakeStanModel.calls[-1] == ("log_density", [0.1], False, False)


def test_cluster_subclasses_keep_jobqueue_cluster_class_configurable(monkeypatch):
    monkeypatch.setattr(PBSClusterBridgeStan, "cluster_class", FakeCluster)

    wrapper = PBSClusterBridgeStan("bernoulli.stan", cluster_kwargs={"queue": "batch"})

    assert wrapper.create_cluster().kwargs == {"queue": "batch"}
