#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=student_bres         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16GB                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus=1               # gpus per node
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)

source /etc/profile
ml torchvision

##################################


python3 create_excel_summary.py -m densenet121 -t Breslow -d Argenciano
python3 create_excel_summary.py -m densenet121 -t Breslow -d ISIC
python3 create_excel_summary.py -m densenet121 -t Breslow -d Polesie
python3 create_excel_summary.py -m densenet121 -t Breslow -d rocio

python3 create_excel_summary.py -m densenet121 -t InSitu -d Argenciano
python3 create_excel_summary.py -m densenet121 -t InSitu -d ISIC
python3 create_excel_summary.py -m densenet121 -t InSitu -d Polesie
python3 create_excel_summary.py -m densenet121 -t InSitu -d rocio

python3 create_excel_summary.py -m densenet121 -t Multiclass -d Argenciano
python3 create_excel_summary.py -m densenet121 -t Multiclass -d ISIC
python3 create_excel_summary.py -m densenet121 -t Multiclass -d Polesie
python3 create_excel_summary.py -m densenet121 -t Multiclass -d rocio




python3 create_excel_summary.py -m resnet50 -t Breslow -d Argenciano
python3 create_excel_summary.py -m resnet50 -t Breslow -d ISIC
python3 create_excel_summary.py -m resnet50 -t Breslow -d Polesie
python3 create_excel_summary.py -m resnet50 -t Breslow -d rocio

python3 create_excel_summary.py -m resnet50 -t InSitu -d Argenciano
python3 create_excel_summary.py -m resnet50 -t InSitu -d ISIC
python3 create_excel_summary.py -m resnet50 -t InSitu -d Polesie
python3 create_excel_summary.py -m resnet50 -t InSitu -d rocio

python3 create_excel_summary.py -m resnet50 -t Multiclass -d Argenciano
python3 create_excel_summary.py -m resnet50 -t Multiclass -d ISIC
python3 create_excel_summary.py -m resnet50 -t Multiclass -d Polesie
python3 create_excel_summary.py -m resnet50 -t Multiclass -d rocio





python3 create_excel_summary.py -m vgg16 -t Breslow -d Argenciano
python3 create_excel_summary.py -m vgg16 -t Breslow -d ISIC
python3 create_excel_summary.py -m vgg16 -t Breslow -d Polesie
python3 create_excel_summary.py -m vgg16 -t Breslow -d rocio

python3 create_excel_summary.py -m vgg16 -t InSitu -d Argenciano
python3 create_excel_summary.py -m vgg16 -t InSitu -d ISIC
python3 create_excel_summary.py -m vgg16 -t InSitu -d Polesie
python3 create_excel_summary.py -m vgg16 -t InSitu -d rocio

python3 create_excel_summary.py -m vgg16 -t Multiclass -d Argenciano
python3 create_excel_summary.py -m vgg16 -t Multiclass -d ISIC
python3 create_excel_summary.py -m vgg16 -t Multiclass -d Polesie
python3 create_excel_summary.py -m vgg16 -t Multiclass -d rocio



# python3 create_excel_summary.py -a Fully_Supervised -m densenet121 -t Multiclass
# python3 create_excel_summary.py -a Fully_Supervised -m vgg16 -t Multiclass
# python3 create_excel_summary.py -a Fully_Supervised -m resnet50 -t Multiclass

# python3 create_excel_summary.py -a Fully_Supervised -m densenet121 -t Breslow
# python3 create_excel_summary.py -a Fully_Supervised -m vgg16 -t Breslow
# python3 create_excel_summary.py -a Fully_Supervised -m resnet50 -t Breslow

# python3 create_excel_summary.py -a Fully_Supervised -m densenet121 -t InSitu
# python3 create_excel_summary.py -a Fully_Supervised -m vgg16 -t InSitu
# python3 create_excel_summary.py -a Fully_Supervised -m resnet50 -t InSitu

################################