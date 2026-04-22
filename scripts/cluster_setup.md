# Multi-laptop MPI Cluster Setup

This is the runbook for executing `train_mpi` and `train_hybrid` across **3 physical laptops** (Phase 7 in the project checklist). The same procedure scales to 2 or N machines.

## Prerequisites

- All laptops on the **same LAN** (Wi-Fi or Ethernet); each can ping the others.
- Identical Linux distro recommended (Ubuntu 24.04 used in this project).
- Same UNIX username on every machine (e.g. `hpc`).
- Same project path on every machine (e.g. `~/hpc-project`) — MPI launches the binary at the *same path* on every node.

## 1. Install MPI on every laptop

```bash
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev build-essential
mpicc --version
mpirun --version   # must be the SAME major version on all nodes
```

## 2. Passwordless SSH from launcher → workers

Pick one laptop as the **launcher** (call it `node0`). From `node0`:

```bash
ssh-keygen -t ed25519                # press enter through prompts
ssh-copy-id hpc@node1
ssh-copy-id hpc@node2
ssh hpc@node1 hostname               # smoke check
ssh hpc@node2 hostname
```

## 3. Replicate the project + dataset

From `node0`:

```bash
rsync -av --exclude='.git' ~/hpc-project/ hpc@node1:~/hpc-project/
rsync -av --exclude='.git' ~/hpc-project/ hpc@node2:~/hpc-project/
```

Make sure `data/processed/cb513/binary/` is present on every node — that path is what `--data` points to.

## 4. Build on every laptop

```bash
ssh hpc@node1 'cd ~/hpc-project && make train_mpi train_hybrid'
ssh hpc@node2 'cd ~/hpc-project && make train_mpi train_hybrid'
make train_mpi train_hybrid          # on node0
```

## 5. Hosts file

Create `scripts/hosts.txt` on `node0` (one line per node, `slots=N` = ranks per node):

```
node0 slots=1
node1 slots=1
node2 slots=1
```

For the hybrid runs, keep `slots=1` per node and let OpenMP take the cores inside each rank.

## 6. Smoke test the cluster

```bash
mpirun --hostfile scripts/hosts.txt -np 3 hostname
# Expected output: node0, node1, node2 (in any order)
```

If you get connection errors, also verify firewalls allow MPI's TCP and that hostnames resolve (`/etc/hosts` if no DNS).

## 7. Real training run — pure MPI

```bash
mpirun --hostfile scripts/hosts.txt -np 3 \
    ~/hpc-project/train_mpi \
        --data ~/hpc-project/data/processed/cb513/binary \
        --epochs 80 --batch 64 --lr 0.01 --seed 42 \
        --hidden1 256 --hidden2 128 \
        --out ~/hpc-project/results/mpi/cluster_3node \
        --verbose 1
```

Sweep configurations:
```bash
for r in 1 2 3; do
    mpirun --hostfile scripts/hosts.txt -np $r \
        ~/hpc-project/train_mpi --data ~/hpc-project/data/processed/cb513/binary \
        --epochs 80 --batch 64 --lr 0.01 --seed 42 \
        --hidden1 256 --hidden2 128 --out ~/hpc-project/results/mpi
done
```

## 8. Real training run — hybrid (MPI + OpenMP)

`MPI ranks × OMP threads ≤ physical cores per laptop` (avoid oversubscription):

```bash
export OMP_PROC_BIND=true OMP_PLACES=cores
mpirun --hostfile scripts/hosts.txt -np 3 \
    ~/hpc-project/train_hybrid --threads 4 \
        --data ~/hpc-project/data/processed/cb513/binary \
        --epochs 80 --batch 64 --lr 0.01 --seed 42 \
        --hidden1 256 --hidden2 128 \
        --out ~/hpc-project/results/hybrid \
        --verbose 1
```

Recommended hybrid sweep (from project plan):
| ranks | threads/rank | total cores |
|-------|--------------|-------------|
| 1     | 1            | 1           |
| 1     | 8            | 8           |
| 1     | 16           | 16          |
| 2     | 8            | 16          |
| 3     | 8            | 24          |

## 9. Collecting results back to launcher

JSON logs are written by rank 0 to its local `--out` directory. To gather them:

```bash
rsync -av hpc@node0:~/hpc-project/results/ ./results_cluster/
```

(Workers don't write logs in the current implementation.)

## Troubleshooting

- **Hangs on MPI_Init / no output:** check firewall (`sudo ufw allow from <subnet>`). On some WSL2 setups OpenMPI's TCP plugin times out; either run on bare-metal Linux or switch to `--mca btl self,vader,tcp --mca btl_tcp_if_include <iface>`.
- **"hostfile not found":** path is relative to the launcher's CWD.
- **"executable not found on node1":** the binary path must be identical on every node (that's why we use `~/hpc-project` everywhere).
- **Different MPI versions across nodes:** apt-installed OpenMPI must match exactly. Mismatched versions → silent hangs at the first collective.
