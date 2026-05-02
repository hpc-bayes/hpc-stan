from __future__ import annotations

import hpc_stan.hpc_bridgestan as bridgestan_module
import hpc_stan.hpc_cmdstanpy as cmdstanpy_module


class RecordingCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scaled_to = None
        self.closed = False

    def scale(self, jobs):
        self.scaled_to = jobs

    def close(self):
        self.closed = True


class RecordingClient:
    def __init__(self, cluster, **kwargs):
        self.cluster = cluster
        self.kwargs = kwargs
        self.closed = False

    def close(self):
        self.closed = True


def test_cmdstanpy_jobqueue_cluster_classes_are_wired():
    expected = {
        cmdstanpy_module.CmdStanPySLURMCluster: cmdstanpy_module.SLURMCluster,
        cmdstanpy_module.CmdStanPyPBSCluster: cmdstanpy_module.PBSCluster,
        cmdstanpy_module.CmdStanPySGECluster: cmdstanpy_module.SGECluster,
        cmdstanpy_module.CmdStanPyLSFCluster: cmdstanpy_module.LSFCluster,
        cmdstanpy_module.CmdStanPyOARCluster: cmdstanpy_module.OARCluster,
        cmdstanpy_module.CmdStanPyMOABCluster: cmdstanpy_module.MoabCluster,
        cmdstanpy_module.CmdStanPyHTCondorCluster: cmdstanpy_module.HTCondorCluster,
    }

    assert {wrapper: wrapper.cluster_class for wrapper in expected} == expected


def test_bridgestan_jobqueue_cluster_classes_are_wired():
    expected = {
        bridgestan_module.SLURMClusterBridgeStan: bridgestan_module.SLURMCluster,
        bridgestan_module.PBSClusterBridgeStan: bridgestan_module.PBSCluster,
        bridgestan_module.SGEClusterBridgeStan: bridgestan_module.SGECluster,
        bridgestan_module.LSFClusterBridgeStan: bridgestan_module.LSFCluster,
        bridgestan_module.OARClusterBridgeStan: bridgestan_module.OARCluster,
        bridgestan_module.MOABClusterBridgeStan: bridgestan_module.MoabCluster,
        bridgestan_module.HTCondorClusterBridgeStan: bridgestan_module.HTCondorCluster,
    }

    assert {wrapper: wrapper.cluster_class for wrapper in expected} == expected


def test_jobqueue_cluster_kwargs_scaling_and_client_kwargs_flow_through(monkeypatch):
    monkeypatch.setattr(cmdstanpy_module.CmdStanPySLURMCluster, "cluster_class", RecordingCluster)

    wrapper = cmdstanpy_module.CmdStanPySLURMCluster(
        "model.stan",
        cluster_kwargs={"queue": "debug", "cores": 2, "memory": "4GB"},
        client_kwargs={"timeout": "30s"},
        n_jobs=4,
        client_class=RecordingClient,
    )

    client = wrapper.get_client()

    assert wrapper.cluster.kwargs == {"queue": "debug", "cores": 2, "memory": "4GB"}
    assert wrapper.cluster.scaled_to == 4
    assert client.cluster is wrapper.cluster
    assert client.kwargs == {"timeout": "30s"}
