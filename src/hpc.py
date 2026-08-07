import os
import socket

def get_hpc_name():
    # 1. Check SLURM environment variables
    if 'SLURM_CLUSTER_NAME' in os.environ:
        return os.environ['SLURM_CLUSTER_NAME']
    if 'SLURM_SUBMIT_HOST' in os.environ:
        return os.environ['SLURM_SUBMIT_HOST']
        
    # 2. Check PBS / Torque environment variables
    if 'PBS_O_HOST' in os.environ:
        return os.environ['PBS_O_HOST']
    if 'PBS_SERVER' in os.environ:
        return os.environ['PBS_SERVER']

    # 3. Check LSF environment variables
    if 'LSF_CLUSTER_NAME' in os.environ:
        return os.environ['LSF_CLUSTER_NAME']
        
    # 4. Check custom HPC environment variables often set by sysadmins
    for var in ['HPC_SYSTEM', 'CLUSTER_NAME', 'SYSTEM_NAME']:
        if var in os.environ:
            return os.environ[var]

    # 5. Fallback to node hostname (e.g., node012.tetralith.nsc.liu.se)
    return socket.gethostname()

print("Current HPC / Node Name:", get_hpc_name())
