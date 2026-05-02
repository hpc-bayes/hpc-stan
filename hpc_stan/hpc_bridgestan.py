"""Dask/jobqueue wrappers for BridgeStan's Python interface."""

from __future__ import annotations

from typing import Any

from bridgestan import StanModel
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

BRIDGESTAN_MODEL_METHODS = (
    "log_density",
    "log_density_gradient",
    "log_density_hessian",
    "log_density_hessian_vector_product",
    "model_info",
    "name",
    "new_rng",
    "param_constrain",
    "param_names",
    "param_num",
    "param_unc_names",
    "param_unc_num",
    "param_unconstrain",
    "param_unconstrain_json",
)


def _build_bridgestan_model(
    stan_file: str,
    data: Any,
    seed: int,
    capture_stan_prints: bool,
    stanc_args: list[str],
    make_args: list[str],
) -> StanModel:
    return StanModel.from_stan_file(
        stan_file,
        data,
        seed=seed,
        capture_stan_prints=capture_stan_prints,
        stanc_args=stanc_args,
        make_args=make_args,
    )


def _run_bridgestan_method(
    stan_file: str,
    data: Any,
    seed: int,
    capture_stan_prints: bool,
    stanc_args: list[str],
    make_args: list[str],
    method_name: str,
    method_args: tuple[Any, ...],
    method_kwargs: dict[str, Any],
) -> Any:
    model = _build_bridgestan_model(
        stan_file,
        data,
        seed,
        capture_stan_prints,
        stanc_args,
        make_args,
    )
    method = getattr(model, method_name)
    return method(*method_args, **method_kwargs)


class HPCBridgeStanBase:
    """Base Dask wrapper around :class:`bridgestan.StanModel`.

    BridgeStan is an in-memory model interface, not an MCMC sampler. This class
    exposes BridgeStan methods such as ``log_density`` and
    ``log_density_gradient`` through Dask while keeping a local ``model`` for
    direct use after ``compile_model``.
    """

    cluster_class: type | None = None

    def __init__(
        self,
        stan_file_path: str,
        model_data: Any | None = None,
        *,
        seed: int = 1234,
        capture_stan_prints: bool = True,
        stanc_args: list[str] | None = None,
        make_args: list[str] | None = None,
        cluster_kwargs: dict[str, Any] | None = None,
        client_kwargs: dict[str, Any] | None = None,
        n_jobs: int | None = None,
        cluster: Any | None = None,
        client: Any | None = None,
        client_class: type = Client,
    ) -> None:
        self.stan_file_path = stan_file_path
        self.model_data = model_data
        self.seed = seed
        self.capture_stan_prints = capture_stan_prints
        self.stanc_args = list(stanc_args or [])
        self.make_args = list(make_args or [])
        self.cluster_kwargs = dict(cluster_kwargs or {})
        self.client_kwargs = dict(client_kwargs or {})
        self.n_jobs = n_jobs
        self.cluster = cluster
        self.client = client
        self.client_class = client_class
        self.model: StanModel | None = None

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

    def compile_model(self) -> StanModel:
        self.model = _build_bridgestan_model(
            self.stan_file_path,
            self.model_data,
            self.seed,
            self.capture_stan_prints,
            self.stanc_args,
            self.make_args,
        )
        return self.model

    def submit(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Submit a BridgeStan model method call to Dask."""

        return self.get_client().submit(
            _run_bridgestan_method,
            self.stan_file_path,
            self.model_data,
            self.seed,
            self.capture_stan_prints,
            self.stanc_args,
            self.make_args,
            method_name,
            args,
            kwargs,
        )

    def __getattr__(self, method_name: str) -> Any:
        if method_name.startswith("_"):
            raise AttributeError(method_name)

        def dask_method(*args: Any, **kwargs: Any) -> Any:
            return self.submit(method_name, *args, **kwargs)

        return dask_method

    def log_density(self, theta_unc: Any, propto: bool = True, jacobian: bool = True) -> Any:
        return self.submit("log_density", theta_unc, propto=propto, jacobian=jacobian)

    def log_density_gradient(self, theta_unc: Any, propto: bool = True, jacobian: bool = True) -> Any:
        return self.submit("log_density_gradient", theta_unc, propto=propto, jacobian=jacobian)

    def log_density_hessian(self, theta_unc: Any, propto: bool = True, jacobian: bool = True) -> Any:
        return self.submit("log_density_hessian", theta_unc, propto=propto, jacobian=jacobian)

    def log_density_hessian_vector_product(self, theta_unc: Any, vector: Any) -> Any:
        return self.submit("log_density_hessian_vector_product", theta_unc, vector)

    def name(self) -> Any:
        return self.submit("name")

    def model_info(self) -> Any:
        return self.submit("model_info")

    def new_rng(self, seed: int) -> Any:
        return self.submit("new_rng", seed)

    def param_unc_num(self) -> Any:
        return self.submit("param_unc_num")

    def param_unc_names(self) -> Any:
        return self.submit("param_unc_names")

    def param_num(self, include_tp: bool = False, include_gq: bool = False) -> Any:
        return self.submit("param_num", include_tp=include_tp, include_gq=include_gq)

    def param_names(self, include_tp: bool = False, include_gq: bool = False) -> Any:
        return self.submit("param_names", include_tp=include_tp, include_gq=include_gq)

    def param_constrain(self, theta_unc: Any, include_tp: bool = False, include_gq: bool = False) -> Any:
        return self.submit("param_constrain", theta_unc, include_tp=include_tp, include_gq=include_gq)

    def param_unconstrain(self, theta: Any) -> Any:
        return self.submit("param_unconstrain", theta)

    def param_unconstrain_json(self, theta_json: Any) -> Any:
        return self.submit("param_unconstrain_json", theta_json)


def _bridge_cluster_type(name: str, cluster_class: type) -> type[HPCBridgeStanBase]:
    return type(name, (HPCBridgeStanBase,), {"cluster_class": cluster_class})


SLURMClusterBridgeStan = _bridge_cluster_type("SLURMClusterBridgeStan", SLURMCluster)
PBSClusterBridgeStan = _bridge_cluster_type("PBSClusterBridgeStan", PBSCluster)
SGEClusterBridgeStan = _bridge_cluster_type("SGEClusterBridgeStan", SGECluster)
LSFClusterBridgeStan = _bridge_cluster_type("LSFClusterBridgeStan", LSFCluster)
OARClusterBridgeStan = _bridge_cluster_type("OARClusterBridgeStan", OARCluster)
MOABClusterBridgeStan = _bridge_cluster_type("MOABClusterBridgeStan", MoabCluster)
HTCondorClusterBridgeStan = _bridge_cluster_type("HTCondorClusterBridgeStan", HTCondorCluster)
