#!/bin/bash
#PBS -l select=1:ncpus=32:mem=32gb
#PBS -l walltime=05:00:00
#PBS -N nanog_sox2_lhs
module load Python/3.12.3-GCCcore-13.3.0

eval "$(conda shell.bash hook)"
conda activate stochastic
cd $PBS_O_WORKDIR || exit
python run_model.py 