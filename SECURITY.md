# Security Policy

## Credential Handling

hpc-stan is designed to run Stan workloads through Dask on clusters that are
already configured. The package should not store cluster credentials, SSH
passwords, scheduler tokens, cloud credentials, or account secrets.

Recommended practices:

- Use SSH keys, SSH agent auth, or site-managed single sign-on where possible.
- Keep passwords and tokens out of source files, notebooks, tests, examples,
  and job scripts committed to git.
- Use environment variables or your institution's secret manager for runtime
  secrets.
- Do not print or log command output that may include credentials.
- Use least-privilege scheduler accounts and short walltimes/resource requests
  for tests.

## SSH Host Keys

The optional SSH utilities reject unknown host keys by default. This helps
avoid accidental connections to the wrong host. If you intentionally need to
connect to a new host, add it to your known hosts file first.

An explicit `allow_unknown_host=True` option exists for controlled development
scenarios, but it should not be used for production cluster access.

## Cluster Setup

hpc-stan does not provision or secure clusters. Users are responsible for
their site's scheduler configuration, network policy, authentication, storage
permissions, and Dask/jobqueue deployment settings.

## Reporting Security Issues

Please report suspected security issues privately to the project maintainer
rather than opening a public issue with exploit details.
