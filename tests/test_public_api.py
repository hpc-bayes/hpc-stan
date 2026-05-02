from __future__ import annotations

import hpc_stan


def test_public_api_exports_cmdstanpy_and_bridgestan_only():
    assert "CmdStanPySLURMCluster" in hpc_stan.__all__
    assert "SLURMClusterBridgeStan" in hpc_stan.__all__
    assert all("PyStan" not in name for name in hpc_stan.__all__)


def test_cmdstanpy_declared_methods_are_forwarded():
    from hpc_stan.hpc_cmdstanpy import BaseClusterCmdStanPy, CMDSTANPY_MODEL_METHODS

    missing = [name for name in CMDSTANPY_MODEL_METHODS if not callable(getattr(BaseClusterCmdStanPy, name, None))]

    assert missing == []


def test_bridgestan_declared_methods_are_forwarded():
    from hpc_stan.hpc_bridgestan import BRIDGESTAN_MODEL_METHODS, HPCBridgeStanBase

    missing = [name for name in BRIDGESTAN_MODEL_METHODS if not callable(getattr(HPCBridgeStanBase, name, None))]

    assert missing == []
