"""Small import shims for unit tests when optional HPC deps are absent."""

from __future__ import annotations

import importlib.util
import sys
import types

CMDSTANPY_AVAILABLE = importlib.util.find_spec("cmdstanpy") is not None
BRIDGESTAN_AVAILABLE = importlib.util.find_spec("bridgestan") is not None
DASK_DISTRIBUTED_AVAILABLE = importlib.util.find_spec("distributed") is not None


class _FakeCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scaled_to = None
        self.closed = False

    def scale(self, jobs):
        self.scaled_to = jobs

    def close(self):
        self.closed = True


class _FakeClient:
    def __init__(self, cluster=None, **kwargs):
        self.cluster = cluster
        self.kwargs = kwargs
        self.closed = False

    def submit(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def gather(self, futures):
        return list(futures)

    def close(self):
        self.closed = True


if not DASK_DISTRIBUTED_AVAILABLE:
    dask = types.ModuleType("dask")
    distributed = types.ModuleType("dask.distributed")
    distributed.Client = _FakeClient
    dask.distributed = distributed
    sys.modules["dask"] = dask
    sys.modules["dask.distributed"] = distributed

if importlib.util.find_spec("dask_jobqueue") is None:
    dask_jobqueue = types.ModuleType("dask_jobqueue")
    for name in [
        "HTCondorCluster",
        "LSFCluster",
        "MoabCluster",
        "OARCluster",
        "PBSCluster",
        "SGECluster",
        "SLURMCluster",
    ]:
        setattr(dask_jobqueue, name, type(name, (_FakeCluster,), {}))
    sys.modules["dask_jobqueue"] = dask_jobqueue

if not CMDSTANPY_AVAILABLE:
    cmdstanpy = types.ModuleType("cmdstanpy")
    cmdstanpy.CmdStanModel = object
    sys.modules["cmdstanpy"] = cmdstanpy

if not BRIDGESTAN_AVAILABLE:
    bridgestan = types.ModuleType("bridgestan")
    bridgestan.StanModel = object
    sys.modules["bridgestan"] = bridgestan
