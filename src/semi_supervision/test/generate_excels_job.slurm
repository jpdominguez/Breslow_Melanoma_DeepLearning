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


python3 get_mean_metrics.py -n 20 -m densenet121 -c 3 -t Multiclass
python3 get_mean_metrics.py -n 30 -m densenet121 -c 3 -t Multiclass
python3 get_mean_metrics.py -n 20 -m densenet121 -c 2 -t InSitu
python3 get_mean_metrics.py -n 30 -m densenet121 -c 2 -t InSitu
python3 get_mean_metrics.py -n 20 -m densenet121 -c 2 -t Breslow
python3 get_mean_metrics.py -n 30 -m densenet121 -c 2 -t Breslow

python3 get_mean_metrics.py -n 20 -m resnet50 -c 3 -t Multiclass
python3 get_mean_metrics.py -n 30 -m resnet50 -c 3 -t Multiclass
python3 get_mean_metrics.py -n 20 -m resnet50 -c 2 -t InSitu
python3 get_mean_metrics.py -n 30 -m resnet50 -c 2 -t InSitu
python3 get_mean_metrics.py -n 20 -m resnet50 -c 2 -t Breslow
python3 get_mean_metrics.py -n 30 -m resnet50 -c 2 -t Breslow


##################################

python3 create_excel_summary.py -m densenet121 -t Multiclass
python3 create_excel_summary.py -m resnet50 -t Multiclass

python3 create_excel_summary.py -m densenet121 -t Breslow
python3 create_excel_summary.py -m resnet50 -t Breslow

python3 create_excel_summary.py -m densenet121 -t InSitu
python3 create_excel_summary.py -m resnet50 -t InSitu

#################################


python3 create_excel_summary_external.py -m densenet121 -t Multiclass
python3 create_excel_summary_external.py -m resnet50 -t Multiclass

python3 create_excel_summary_external.py -m densenet121 -t Breslow
python3 create_excel_summary_external.py -m resnet50 -t Breslow

python3 create_excel_summary_external.py -m densenet121 -t InSitu
python3 create_excel_summary_external.py -m resnet50 -t InSitu


#################################

python3 create_excel_mean_std.py -t Multiclass
python3 create_excel_mean_std.py -t Breslow
python3 create_excel_mean_std.py -t InSitu

python3 create_excel_mean_std_external.py -t Multiclass
python3 create_excel_mean_std_external.py -t Breslow
python3 create_excel_mean_std_external.py -t InSitu


#################################

python3 create_bests_report.py
python3 create_bests_report_external.py
