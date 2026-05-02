# hpc-stan

A Dask-powered HPC/cloud friendly interface for Stan workflows using
CmdStanPy and BridgeStan.

The goal is to keep the CmdStanPy and BridgeStan APIs recognizable while
adding a Dask execution layer and `dask-jobqueue` cluster support. PyStan is
not supported because the upstream project was archived.

## What This Package Does

hpc-stan helps you run Stan workloads on an already configured Dask or
`dask-jobqueue` cluster.

It can:

- Wrap CmdStanPy model methods so sampling, optimization, variational
  inference, generated quantities, and related methods run through Dask.
- Wrap BridgeStan model methods so log density, gradient, Hessian, parameter
  metadata, and transform calls run through Dask.
- Create `dask-jobqueue` cluster objects for supported schedulers using the
  scheduler settings you pass in `cluster_kwargs`.
- Scale the Dask jobqueue cluster with `n_jobs`.
- Let you use local Dask clients in tests and development, then switch to an
  HPC scheduler wrapper in production.

It cannot:

- Install, configure, or administer SLURM, PBS, SGE, LSF, OAR, Moab, or
  HTCondor.
- Create cloud infrastructure, login nodes, queues, accounts, allocations, or
  scheduler permissions.
- Install CmdStan, BridgeStan build dependencies, compilers, MPI, modules, or
  site-specific environment scripts on your cluster.
- Decide the right queue, walltime, memory, cores, project account, module
  loads, or scratch paths for your site.
- Replace your cluster administrator's policies or your site's Dask/jobqueue
  configuration.

In short: bring a working cluster and working Stan toolchains; hpc-stan helps
route the work through Dask.

## Supported Backends

- CmdStanPy for CmdStan sampling, optimization, variational inference,
  Pathfinder, Laplace sampling, generated quantities, diagnostics, and other
  `CmdStanModel` methods.
- BridgeStan for in-memory model operations such as parameter metadata,
  constrain/unconstrain, log density, gradients, Hessians, and model
  information.

Public backend methods are forwarded through Dask. Named wrapper methods are
provided for the major APIs, and `submit(method_name, *args, **kwargs)` can be
used for any backend method.

## Supported Schedulers

The package uses `dask-jobqueue` cluster classes for:

- SLURM
- PBS
- SGE
- LSF
- OAR
- Moab
- HTCondor

Each scheduler has a CmdStanPy wrapper and a BridgeStan wrapper. For example,
`CmdStanPySLURMCluster` uses `dask_jobqueue.SLURMCluster`, and
`PBSClusterBridgeStan` uses `dask_jobqueue.PBSCluster`.

## Installation

Using pip:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Using conda:

```bash
conda env create -f environment.yml
conda activate hpcstan_env
python -m pip install -e .
```

CmdStanPy requires a CmdStan installation before CmdStan-backed tests or runs
can execute:

```bash
python -m cmdstanpy.install_cmdstan --cores 2
```

On managed HPC systems, you may need to use your site's compiler modules,
filesystem paths, scheduler accounts, and environment setup instead of the
default local CmdStan installer.

## CmdStanPy Example

```python
from hpc_stan import CmdStanPySLURMCluster

stan = CmdStanPySLURMCluster(
    "model.stan",
    cluster_kwargs={
        "cores": 2,
        "memory": "4GB",
        "walltime": "01:00:00",
        "queue": "regular",
    },
    n_jobs=2,
)

fits = stan.sample(
    n_tasks=2,
    data="data.json",
    chains=1,
    iter_warmup=500,
    iter_sampling=500,
    show_progress=False,
)
```

Other CmdStanPy methods follow the same pattern:

```python
mle_future = stan.optimize(data="data.json", algorithm="LBFGS")
vb_future = stan.variational(data="data.json", algorithm="meanfield")
```

These return Dask futures for single backend method calls. Use the client to
gather them when needed.

## BridgeStan Example

```python
from hpc_stan import SLURMClusterBridgeStan

bridge = SLURMClusterBridgeStan(
    "model.stan",
    "data.json",
    cluster_kwargs={"cores": 1, "memory": "2GB", "walltime": "00:30:00"},
    n_jobs=2,
)

lp_future = bridge.log_density([0.0], propto=False, jacobian=True)
grad_future = bridge.log_density_gradient([0.0])
```

## Tests

The test suite includes:

- Dask wrapper unit tests.
- `dask-jobqueue` scheduler wiring tests.
- CmdStanPy behavior tests ported from the upstream CmdStanPy test style and
  run through the hpc-stan Dask wrapper.
- BridgeStan behavior tests ported from the upstream BridgeStan Python test
  style and run through the hpc-stan Dask wrapper.

The Stan-backed tests use embedded models in `tests/models`.

```bash
python -m pytest
```

If CmdStanPy, BridgeStan, CmdStan, or `dask.distributed` are unavailable, the
corresponding Stan-backed tests skip with an explicit reason.

## Docker

Build a lightweight development/test image:

```bash
docker build -t hpc-stan .
```

Build with CmdStan included:

```bash
docker build --build-arg INSTALL_CMDSTAN=true -t hpc-stan .
```

Run the test suite:

```bash
docker run --rm hpc-stan
```

## Security Notes

hpc-stan does not need to store scheduler or SSH credentials. Prefer existing
Dask/jobqueue configuration, SSH agent forwarding, short-lived credentials,
and site-managed authentication.

For the optional SSH helpers in `hpc_stan.utils`:

- Prefer SSH keys or agent auth over passwords.
- Unknown SSH host keys are rejected by default.
- Passwords are accepted as function arguments or prompted with `getpass`, but
  should not be committed, logged, or stored in config files.
- Remote stdout/stderr are returned to the caller and are not printed unless
  `stream_output=True`.
- Keep secrets in environment variables or your site's secret manager, not in
  notebooks, tests, examples, or job scripts committed to git.

### Optional SSH Helpers

Most users should connect to their HPC resources through the normal
`dask-jobqueue` path shown above. The helpers in `hpc_stan.utils` are only for
cases where you explicitly need SSH tunneling or remote script submission.

Forward a Dask dashboard with key-based auth:

```python
from hpc_stan.utils import forward_dask_dashboard

server = forward_dask_dashboard(
    "login.cluster.edu",
    username="my_user",
    key_filename="~/.ssh/id_ed25519",
)

# Later, when finished:
server.stop()
```

Submit a script to a known host without printing remote output:

```python
from hpc_stan.utils import submit_to_cluster

stdout, stderr = submit_to_cluster(
    "login.cluster.edu",
    username="my_user",
    key_filename="~/.ssh/id_ed25519",
    script_path="run_stan_job.py",
    remote_script_path="run_stan_job.py",
    command="python run_stan_job.py",
)
```

Unknown SSH host keys are rejected by default. Add the cluster login host to
your known hosts before using these helpers.
