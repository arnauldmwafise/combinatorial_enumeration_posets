#!/bin/bash
# =============================================================================
# SLURM BATCH JOB EXECUTION SPECIFICATION SCRIPT
# =============================================================================
#SBATCH --job-name=poset_operad_growth       # Unique job identifier
#SBATCH --output=docs/cluster_output_%j.log   # Combined stdout/stderr tracking destination
#SBATCH --partition=gpu                       # Route execution explicitly to the GPU cluster pool
#SBATCH --gres=gpu:1                          # Allocate 1 dedicated enterprise GPU node (e.g. A100)
#SBATCH --nodes=1                             # Bind job tasks to a single compute node
#SBATCH --ntasks=1                            # Single master process instance
#SBATCH --cpus-per-task=8                     # Provide supplementary multithreaded CPU cores
#SBATCH --mem=32G                             # Allocate 32 Gigabytes of system memory capacity
#SBATCH --time=12:00:00                       # Set hard 12-hour compute walltime fence

set -e

echo "=== [HPC LOG] Initializing Automated Cluster Computing Pipeline ==="
echo "Date/Time       : \06/25/2026 13:11:38"
echo "Execution Node  : \"
echo "Allocated GPU   : \"
echo "------------------------------------------------------------------------"

module purge
module load python/3.12
module load cuda/12.1

export PYTHONPATH="src:\"

python -m pip install --user --upgrade pip
python -m pip install --user -r requirements.txt

python -m pytest -v tests/
python verify_ops.py
python benchmark_poset.py
python plot_benchmark.py

echo "------------------------------------------------------------------------"
echo "=== [HPC LOG] Cluster Processing Phase Completed Successfully ==="
echo "End Date/Time   : \06/25/2026 13:11:38"
