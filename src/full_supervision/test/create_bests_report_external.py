import pandas as pd
import os
import numpy as np
import sys

def create_dir(directory):
	if not os.path.isdir(directory):
		try:
			os.mkdir(directory)
		except OSError:
			print ("Creation of the directory %s failed" % directory)
		else:
			print ("Successfully created the directory %s " % directory)

PATH_OUT = '/home/jpdominguez/projects/BreslowTotal/src/full_supervision/test/summary/'
create_dir(PATH_OUT)


import argparse
argv = sys.argv[1:]
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-a', '--APPROACH', help='approach to use',type=str, default='Fully_Supervised')

args = parser.parse_args(argv)
APPROACH = args.APPROACH




N_EXPS = ['10', '20', '30']
MODELS = ['densenet121', 'resnet50', 'vgg16']
TASKS = ['Multiclass', 'InSitu', 'Breslow']


pd_row = pd.DataFrame()


legend = np.array(['TASK', 'MODEL', 'N_EXP', 'acc_balanced', 'acc', 'f1_score', 'kappa_score', 'precision', 'roc_auc_score', 'recall', 'specificity'], dtype=object)

# pd_row = pd.concat([pd_row, pd.DataFrame([MODEL])], ignore_index=True)
pd_row = pd.concat([pd_row, pd.DataFrame(legend.reshape((1, 11)))], ignore_index=True)

for TASK in TASKS:
    if TASK == 'Multiclass':
        metrics = ['acc_balanced', 'acc', 'f1_score', 'kappa_score', 'precision', 'roc_auc']
        reshape_size = (1, 9)
    elif TASK == 'InSitu' or TASK == 'Breslow':
        metrics = ['acc_balanced', 'acc', 'f1_score', 'kappa_score', 'precision', 'roc_auc', 'recall', 'specificity']
        reshape_size = (1, 11)


    best_mean = []
    best_std = []
    best_n_exp = 0
    nest_model = ''
    for MODEL in MODELS:    
        for N_EXP in N_EXPS:

            PATH = '/home/jpdominguez/projects/BreslowTotal/src/full_supervision/train/models/' + MODEL + '/' + TASK + '/N_EXP_' + N_EXP + '/'

            results_per_fold = [[] for i in range(5)]
            
            for i in range(5):
                path_local = PATH + 'Fold_' + str(i) + '/checkpoints/'
                N_EXP_local = []
                N_EXP_local.append('Fold_' + str(i))
                for metric in metrics:
                    local = pd.read_csv(path_local + 'external_test_' + metric + '.csv', header=None)
                    N_EXP_local.append("{:.4f}".format(local[0].values[0]))
                    results_per_fold[i].append(local[0].values[0])
                    
                N_EXP_local = np.array(N_EXP_local, dtype=object)
            
            mean_results = []
            std_results = []
            for i in range(len(results_per_fold[0])):
                results_i_per_fold = [results_per_fold[j][i] for j in range(5)]
                results_i_per_fold = np.array(results_i_per_fold)
                mean_results_i = "{:.4f}".format(np.mean(results_i_per_fold))
                mean_results.append(mean_results_i)
                std_results_i = "{:.4f}".format(np.std(results_i_per_fold))
                std_results.append(std_results_i)
            if best_mean == [] or float(mean_results[0]) > float(best_mean[0]):
                best_mean = mean_results
                best_std = std_results
                best_n_exp = N_EXP
                best_model = MODEL
        

    combined = []
    for i in range(len(best_mean)):
        combined.append(best_mean[i] + ' ± ' + best_std[i])
    combined = [TASK, best_model, best_n_exp] + combined
    combined = np.array(combined, dtype=object).reshape(reshape_size)
    pd_row = pd.concat([pd_row, pd.DataFrame(combined)], ignore_index=True)
 
pd_row.to_excel(PATH_OUT + 'best_results_external.xlsx', header=False, index=False)



# combined = []
# for i in range(len(mean_results)):
#     combined.append(mean_results[i] + ' ± ' + std_results[i])
# combined = ['N_EXP_' + N_EXP] + combined
# combined = np.array(combined, dtype=object).reshape(reshape_size)