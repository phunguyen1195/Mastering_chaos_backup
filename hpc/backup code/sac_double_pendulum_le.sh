#!/bin/bash



#SBATCH --job-name=Double_pendulum_le   # job name
#SBATCH --output=log/lorenz/Lorenz-%j.out # output log file
#SBATCH --error=log/lorenz/Lorenz-%j.err  # error file
#SBATCH --time=24:00:00  # 5 hour of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu # GPU partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --mem-per-cpu=1000
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu     # Request 1 GPU
#SBATCH --mail-user=phu.c.nguyen@sjsu.edu
#SBATCH --mail-type=END

python stablebaselines3_double_pendulum_le_sac.py