# hpc-stan

A Dask-powered HPC/cloud friendly interface for Stan workflows using
CmdStanPy and BridgeStan.

PyStan is not part of the supported API because the upstream project has been
archived. CmdStanPy is used for sampling/optimization workflows, and
BridgeStan is used for in-memory model operations such as log density and
gradient evaluation.

## Supported schedulers

The wrappers use `dask-jobqueue` cluster classes:

- SLURM
- PBS
- SGE
- LSF
- OAR
- Moab
- HTCondor

## Example

```python
from hpc_stan import CmdStanPySLURMCluster

stan = CmdStanPySLURMCluster(
    "model.stan",
    cluster_kwargs={"cores": 2, "memory": "4GB", "walltime": "01:00:00"},
    n_jobs=2,
)

fits = stan.sample(
    n_tasks=2,
    data="data.json",
    chains=1,
    iter_warmup=500,
    iter_sampling=500,
)
```

## Tests

The test suite includes unit coverage for the Dask wrapper behavior and Stan
program tests that compile and run `tests/models/bernoulli.stan` through
CmdStanPy and BridgeStan when those toolchains are installed.

```bash
python -m pytest
```
