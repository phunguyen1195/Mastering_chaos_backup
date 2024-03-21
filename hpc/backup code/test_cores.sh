#!/bin/bash



#SBATCH --job-name=precal_le   # job name
#SBATCH --output=log/lorenz/precal-%j.out # output log file
#SBATCH --error=log/lorenz/precal-%j.err  # error file
#SBATCH --time=24:00:00  # 5 hour of wall time
#SBATCH --nodes=2        # 1 GPU node
#SBATCH --ntasks=2      # 1 CPU core to drive GPU
#SBATCH --cpus-per-task=50
#SBATCH --mail-user=phu.c.nguyen@sjsu.edu
#SBATCH --mail-type=END

python test_cores.py