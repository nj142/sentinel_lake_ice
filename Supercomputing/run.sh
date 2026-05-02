#!/bin/bash
#SBATCH --ntasks=25
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=freezeup_s2
#SBATCH --output=/hpc/home/nj142/Output/Sentinel_DCC_%j.out

module purg
module load OpenMPI/4.1.6 

source /hpc/home/nj142/miniconda3/etc/profile.d/conda.sh
conda activate python311

mpirun -n 25 python ~/Scripts/Sentinel_DCC.py --workers 15
