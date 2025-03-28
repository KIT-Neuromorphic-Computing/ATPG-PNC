#!/usr/bin/env bash

#SBATCH --job-name=atpg_9_12
#SBATCH --partition=single
#SBATCH --ntasks-per-node=10
#SBATCH --time=72:00:00
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.o%j.%N

python atpg_run.py --DATASET 5 &
python atpg_run.py --DATASET 6 &
python atpg_run.py --DATASET 7 &
python atpg_run.py --DATASET 8 &
python atpg_run.py --DATASET 9 &
python atpg_run.py --DATASET 10 &
python atpg_run.py --DATASET 11 &
python atpg_run.py --DATASET 12 &

wait
