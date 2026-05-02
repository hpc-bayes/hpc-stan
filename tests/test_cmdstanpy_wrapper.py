from __future__ import annotations

from hpc_stan.hpc_cmdstanpy import BaseClusterCmdStanPy, CmdStanPySLURMCluster


class ImmediateClient:
    def __init__(self):
        self.closed = False

    def submit(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def gather(self, futures):
        return list(futures)

    def close(self):
        self.closed = True


class FakeCmdStanModel:
    calls = []

    def __init__(self, stan_file, **kwargs):
        self.stan_file = stan_file
        self.model_kwargs = kwargs

    def sample(self, **kwargs):
        self.calls.append(("sample", self.stan_file, self.model_kwargs, kwargs))
        return {"method": "sample", "stan_file": self.stan_file, "kwargs": kwargs}

    def optimize(self, **kwargs):
        self.calls.append(("optimize", self.stan_file, self.model_kwargs, kwargs))
        return {"method": "optimize", "kwargs": kwargs}


class FakeCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scaled_to = None

    def scale(self, jobs):
        self.scaled_to = jobs


def test_sample_dispatches_cmdstanpy_sample_through_dask(monkeypatch):
    import hpc_stan.hpc_cmdstanpy as module

    FakeCmdStanModel.calls = []
    monkeypatch.setattr(module, "CmdStanModel", FakeCmdStanModel)

    wrapper = BaseClusterCmdStanPy(
        "bernoulli.stan",
        model_kwargs={"compile": False},
        client=ImmediateClient(),
    )

    results = wrapper.sample(n_tasks=2, data="bernoulli.json", chains=1)

    assert results == [
        {"method": "sample", "stan_file": "bernoulli.stan", "kwargs": {"data": "bernoulli.json", "chains": 1}},
        {"method": "sample", "stan_file": "bernoulli.stan", "kwargs": {"data": "bernoulli.json", "chains": 1}},
    ]
    assert FakeCmdStanModel.calls == [
        ("sample", "bernoulli.stan", {"compile": False}, {"data": "bernoulli.json", "chains": 1}),
        ("sample", "bernoulli.stan", {"compile": False}, {"data": "bernoulli.json", "chains": 1}),
    ]


def test_cmdstanpy_cluster_builds_client_and_scales_jobs(monkeypatch):
    import hpc_stan.hpc_cmdstanpy as module

    monkeypatch.setattr(CmdStanPySLURMCluster, "cluster_class", FakeCluster)
    monkeypatch.setattr(module, "CmdStanModel", FakeCmdStanModel)

    created_clients = []

    class RecordingClient(ImmediateClient):
        def __init__(self, cluster, **kwargs):
            super().__init__()
            self.cluster = cluster
            self.kwargs = kwargs
            created_clients.append(self)

    wrapper = CmdStanPySLURMCluster(
        "bernoulli.stan",
        cluster_kwargs={"queue": "debug"},
        client_kwargs={"timeout": "30s"},
        n_jobs=3,
        client_class=RecordingClient,
    )

    future = wrapper.optimize(data="bernoulli.json")

    assert future == {"method": "optimize", "kwargs": {"data": "bernoulli.json"}}
    assert wrapper.cluster.kwargs == {"queue": "debug"}
    assert wrapper.cluster.scaled_to == 3
    assert created_clients[0].kwargs == {"timeout": "30s"}
