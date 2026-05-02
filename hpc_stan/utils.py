from getpass import getpass
from pathlib import Path


def forward_dask_dashboard(
    cluster_address,
    remote_port=8787,
    local_port=8787,
    username=None,
    password=None,
    key_filename=None,
):
    from sshtunnel import SSHTunnelForwarder

    username = username if username else input("Enter your username: ")
    if password is None and key_filename is None:
        password = getpass("Enter your SSH password, or Ctrl-C to use key/agent auth instead: ")

    server = SSHTunnelForwarder(
        cluster_address,
        ssh_username=username,
        ssh_password=password,
        ssh_pkey=key_filename,
        remote_bind_address=('localhost', remote_port),
        local_bind_address=('localhost', local_port)
    )

    server.start()
    print(f"Dashboard is being served locally at localhost:{local_port}")

    return server


def submit_to_cluster(
    cluster_address,
    username=None,
    password=None,
    key_filename=None,
    script_path=None,
    remote_script_path="remote_python_script.py",
    command=None,
    allow_unknown_host=False,
    stream_output=False,
):
    import paramiko

    username = username if username else input("Enter your username: ")
    if password is None and key_filename is None:
        password = getpass("Enter your SSH password, or Ctrl-C to use key/agent auth instead: ")

    local_script = Path(script_path or "your_python_script.py").expanduser()
    if not local_script.exists():
        raise FileNotFoundError(f"Script not found: {local_script}")

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    if allow_unknown_host:
        ssh.set_missing_host_key_policy(paramiko.WarningPolicy())
    else:
        ssh.set_missing_host_key_policy(paramiko.RejectPolicy())

    try:
        ssh.connect(
            cluster_address,
            username=username,
            password=password,
            key_filename=key_filename,
            look_for_keys=True,
            allow_agent=True,
        )

        sftp = ssh.open_sftp()
        try:
            sftp.put(str(local_script), remote_script_path)
        finally:
            sftp.close()

        remote_command = command or f"python {remote_script_path}"
        _, stdout, stderr = ssh.exec_command(remote_command)
        stdout_text = stdout.read().decode()
        stderr_text = stderr.read().decode()

        if stream_output:
            print(stdout_text)
            print(stderr_text)

        return stdout_text, stderr_text
    finally:
        ssh.close()
