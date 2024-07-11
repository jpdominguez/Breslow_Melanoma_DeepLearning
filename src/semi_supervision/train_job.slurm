#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=semi_supervised_training         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64GB                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus=a100               # gpus per node
#SBATCH --time=96:00:00          # total run time limit (HH:MM:SS)

source /etc/profile
ml torchvision


# python3 train_student.py -n 20 -b 32 -m resnet50 -t InSitu
# python3 train_student.py -n 30 -b 32 -m resnet50 -t Breslow
# python3 train_student.py -n 30 -b 32 -m densenet121 -t InSitu
# python3 train_student.py -n 30 -b 32 -m densenet121 -t Breslow
# python3 train_student.py -n 30 -b 32 -m densenet121 -t Multiclass

python3 train_student.py -n 20 -b 32 -m resnet50 -t Breslow
python3 train_student.py -n 30 -b 32 -m resnet50 -t InSitu
python3 train_student.py -n 30 -b 32 -m resnet50 -t Multiclass


# for model in 'densenet121' 'resnet50' 'vgg16'
# do
#     for task in 'Breslow' 'InSitu' 'Multiclass'
#     do
#         for n_exp in 10 20 30
#         do
#             python3 train_student.py -n $n_exp -b 32 -m $model -t $task
#         done
#     done
# done