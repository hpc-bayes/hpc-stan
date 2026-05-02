import os
from getpass import getpass


def forward_dask_dashboard(cluster_address, remote_port=8787, local_port=8787, username=None, password=None):
    from sshtunnel import SSHTunnelForwarder

    username = username if username else input("Enter your username: ")
    password = password if password else getpass("Enter your password: ")

    # Note: by default, Dask's dashboard is served on port 8787
    server = SSHTunnelForwarder(
        cluster_address,
        ssh_username=username,
        ssh_password=password,
        remote_bind_address=('localhost', remote_port),
        local_bind_address=('localhost', local_port)
    )

    server.start()
    print(f"Dashboard is being served locally at localhost:{local_port}")

    # It's important to remember to close the server when it's no longer needed
    # server.stop()


def submit_to_cluster(cluster_address, username=None, password=None):
    import paramiko

    username = username if username else input("Enter your username: ")
    password = password if password else getpass("Enter your password: ")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(cluster_address, username=username, password=password)

    # Assuming the Python script is located in the same directory as this Python file
    script_path = os.path.join(os.getcwd(), 'your_python_script.py')
    sftp = ssh.open_sftp()
    sftp.put(script_path, 'remote_python_script.py')
    sftp.close()

    # Modify the command as per your cluster's requirements
    command = "python remote_python_script.py"

    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read().decode())
    print(stderr.read().decode())

    ssh.close()
