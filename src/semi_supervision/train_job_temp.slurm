#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=SS_temp_train         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64GB                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus=a100               # gpus per node
#SBATCH --time=96:00:00          # total run time limit (HH:MM:SS)

source /etc/profile
ml torchvision



python3 train_student.py -n 30 -b 128 -m densenet121 -t Multiclass -l 0.0001
