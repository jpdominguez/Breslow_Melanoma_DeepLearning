#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=annotator         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64GB                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus=1               # gpus per node
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

source /etc/profile
ml torchvision


python3 Annotator.py -n 10 -b 64 -m densenet121 -f majority -t Multiclass
python3 Annotator.py -n 20 -b 64 -m densenet121 -f majority -t Multiclass
python3 Annotator.py -n 30 -b 64 -m densenet121 -f majority -t Multiclass
python3 Annotator.py -n 10 -b 64 -m densenet121 -f majority -t InSitu
python3 Annotator.py -n 20 -b 64 -m densenet121 -f majority -t InSitu
python3 Annotator.py -n 30 -b 64 -m densenet121 -f majority -t InSitu
python3 Annotator.py -n 10 -b 64 -m densenet121 -f majority -t Breslow
python3 Annotator.py -n 20 -b 64 -m densenet121 -f majority -t Breslow
python3 Annotator.py -n 30 -b 64 -m densenet121 -f majority -t Breslow

python3 Annotator.py -n 10 -b 64 -m resnet50 -f majority -t Multiclass
python3 Annotator.py -n 20 -b 64 -m resnet50 -f majority -t Multiclass
python3 Annotator.py -n 30 -b 64 -m resnet50 -f majority -t Multiclass
python3 Annotator.py -n 10 -b 64 -m resnet50 -f majority -t InSitu
python3 Annotator.py -n 20 -b 64 -m resnet50 -f majority -t InSitu
python3 Annotator.py -n 30 -b 64 -m resnet50 -f majority -t InSitu
python3 Annotator.py -n 10 -b 64 -m resnet50 -f majority -t Breslow
python3 Annotator.py -n 20 -b 64 -m resnet50 -f majority -t Breslow
python3 Annotator.py -n 30 -b 64 -m resnet50 -f majority -t Breslow

python3 Annotator.py -n 10 -b 64 -m vgg16 -f majority -t Multiclass
python3 Annotator.py -n 20 -b 64 -m vgg16 -f majority -t Multiclass
python3 Annotator.py -n 30 -b 64 -m vgg16 -f majority -t Multiclass
python3 Annotator.py -n 10 -b 64 -m vgg16 -f majority -t InSitu
python3 Annotator.py -n 20 -b 64 -m vgg16 -f majority -t InSitu
python3 Annotator.py -n 30 -b 64 -m vgg16 -f majority -t InSitu
python3 Annotator.py -n 10 -b 64 -m vgg16 -f majority -t Breslow
python3 Annotator.py -n 20 -b 64 -m vgg16 -f majority -t Breslow
python3 Annotator.py -n 30 -b 64 -m vgg16 -f majority -t Breslow