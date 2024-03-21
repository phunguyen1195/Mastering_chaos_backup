#!/bin/bash
#SBATCH --job-name=precal_le   # job name
##SBATCH --output=log/lorenz/precal-%j.out # output log file
#SBATCH --output=precal-%j.out # output log file
##SBATCH --error=log/lorenz/precal-%j.err  # error file
#SBATCH --error=precal-%j.err  # error file
#SBATCH --time=24:00:00  # 5 hour of wall time
#SBATCH --nodes=2        # 1 GPU node
#SBATCH --cpus-per-task=50     # 1 CPU core to drive GPU
##SBATCH --mail-user=phu.c.nguyen@sjsu.edu
#SBATCH --mail-user=/dev/null
#SBATCH --mail-type=END

# module load openmpi-4.1.4-gcc-12.2.0-naunzth


mpirun --oversubscribe python /home/015970994/masterchaos/precal_le/precal_double_pendulum_ver2.py
