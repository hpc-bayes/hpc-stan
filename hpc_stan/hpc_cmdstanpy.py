"""Dask/jobqueue wrappers for CmdStanPy."""

from __future__ import annotations

from typing import Any, Iterable

from cmdstanpy import CmdStanModel
from dask.distributed import Client
from dask_jobqueue import (
    HTCondorCluster,
    LSFCluster,
    MoabCluster,
    OARCluster,
    PBSCluster,
    SGECluster,
    SLURMCluster,
)

from hpc_stan.utils import submit_to_cluster

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


def _run_cmdstanpy_method(
    stan_file: str,
    method_name: str,
    method_args: tuple[Any, ...],
    model_kwargs: dict[str, Any],
    method_kwargs: dict[str, Any],
) -> Any:
    model = CmdStanModel(stan_file=stan_file, **model_kwargs)
    method = getattr(model, method_name)
    return method(*method_args, **method_kwargs)


class BaseClusterCmdStanPy:
    """Base Dask wrapper around :class:`cmdstanpy.CmdStanModel`.

    The wrapper keeps CmdStanPy calls recognizable. For example,
    ``cluster.sample(data=...)`` dispatches ``CmdStanModel.sample(data=...)``
    through Dask, while ``cluster.submit("optimize", data=...)`` dispatches any
    other CmdStanPy model method.
    """

    cluster_class: type | None = None

    def __init__(
        self,
        stan_file: str,
        *,
        cluster_kwargs: dict[str, Any] | None = None,
        client_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        n_jobs: int | None = None,
        cluster: Any | None = None,
        client: Any | None = None,
        client_class: type = Client,
    ) -> None:
        self.stan_file = stan_file
        self.cluster_kwargs = dict(cluster_kwargs or {})
        self.client_kwargs = dict(client_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.n_jobs = n_jobs
        self.cluster = cluster
        self.client = client
        self.client_class = client_class

    def create_cluster(self) -> Any:
        if self.cluster_class is None:
            raise ValueError("cluster_class must be set by a subclass.")
        return self.cluster_class(**self.cluster_kwargs)

    def get_client(self) -> Any:
        if self.client is None:
            if self.cluster is None:
                self.cluster = self.create_cluster()
            if self.n_jobs is not None and hasattr(self.cluster, "scale"):
                self.cluster.scale(jobs=self.n_jobs)
            self.client = self.client_class(self.cluster, **self.client_kwargs)
        return self.client

    def close(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()
        if self.cluster is not None and hasattr(self.cluster, "close"):
            self.cluster.close()

    def run_on_cluster(self, cluster_address: str, username: str | None = None, password: str | None = None) -> None:
        submit_to_cluster(cluster_address, username, password)

    def submit(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Submit a single CmdStanPy model method call to Dask."""

        return self.get_client().submit(
            _run_cmdstanpy_method,
            self.stan_file,
            method_name,
            args,
            self.model_kwargs,
            kwargs,
        )

    def __getattr__(self, method_name: str) -> Any:
        if method_name.startswith("_"):
            raise AttributeError(method_name)

        def dask_method(*args: Any, **kwargs: Any) -> Any:
            return self.submit(method_name, *args, **kwargs)

        return dask_method

    def map(
        self,
        method_name: str,
        calls: Iterable[tuple[tuple[Any, ...], dict[str, Any]]],
    ) -> list[Any]:
        """Map many CmdStanPy model method calls over Dask.

        Each item in ``calls`` is ``((positional_args,), {keyword_args})``.
        """

        futures = [
            self.submit(method_name, *method_args, **method_kwargs)
            for method_args, method_kwargs in calls
        ]
        return self.get_client().gather(futures)

    def sample(self, n_tasks: int = 1, **kwargs: Any) -> list[Any]:
        """Run ``CmdStanModel.sample`` one or more times on the Dask cluster."""

        calls = [tuple(), kwargs]
        return self.map("sample", [(calls[0], calls[1]) for _ in range(n_tasks)])

    def optimize(self, **kwargs: Any) -> Any:
        return self.submit("optimize", **kwargs)

    def variational(self, **kwargs: Any) -> Any:
        return self.submit("variational", **kwargs)

    def pathfinder(self, **kwargs: Any) -> Any:
        return self.submit("pathfinder", **kwargs)

    def generate_quantities(self, **kwargs: Any) -> Any:
        return self.submit("generate_quantities", **kwargs)

    def laplace_sample(self, **kwargs: Any) -> Any:
        return self.submit("laplace_sample", **kwargs)

    def diagnose(self, **kwargs: Any) -> Any:
        return self.submit("diagnose", **kwargs)

    def log_prob(self, **kwargs: Any) -> Any:
        return self.submit("log_prob", **kwargs)

    def compile(self, **kwargs: Any) -> Any:
        return self.submit("compile", **kwargs)

    def code(self, **kwargs: Any) -> Any:
        return self.submit("code", **kwargs)

    def exe_info(self, **kwargs: Any) -> Any:
        return self.submit("exe_info", **kwargs)

    def src_info(self, **kwargs: Any) -> Any:
        return self.submit("src_info", **kwargs)

    def format(self, **kwargs: Any) -> Any:
        return self.submit("format", **kwargs)

    def run_stan_model_cmdstan(self, n_tasks: int = 10, **sample_kwargs: Any) -> list[Any]:
        """Backward-compatible alias for the original API."""

        return self.sample(n_tasks=n_tasks, **sample_kwargs)

    def run_stan_model_cmdstanpy(self, n_tasks: int = 10, **sample_kwargs: Any) -> list[Any]:
        """Backward-compatible alias for callers that used CmdStanPy naming."""

        return self.sample(n_tasks=n_tasks, **sample_kwargs)


def _cluster_type(cluster_class: type) -> type[BaseClusterCmdStanPy]:
    return type(
        f"CmdStanPy{cluster_class.__name__.removesuffix('Cluster')}Cluster",
        (BaseClusterCmdStanPy,),
        {"cluster_class": cluster_class},
    )


CmdStanPySLURMCluster = _cluster_type(SLURMCluster)
CmdStanPyPBSCluster = _cluster_type(PBSCluster)
CmdStanPySGECluster = _cluster_type(SGECluster)
CmdStanPyLSFCluster = _cluster_type(LSFCluster)
CmdStanPyOARCluster = _cluster_type(OARCluster)
CmdStanPyMOABCluster = _cluster_type(MoabCluster)
CmdStanPyHTCondorCluster = _cluster_type(HTCondorCluster)

# Historical names kept for compatibility.
CmdStanSLURMCluster = CmdStanPySLURMCluster
CmdStanPBSCluster = CmdStanPyPBSCluster
CmdStanSGECluster = CmdStanPySGECluster
CmdStanLSFCluster = CmdStanPyLSFCluster
CmdStanOARCluster = CmdStanPyOARCluster
CmdStanMOABCluster = CmdStanPyMOABCluster
CmdStanHTCondorCluster = CmdStanPyHTCondorCluster
