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

PATH_OUT = '/home/jpdominguez/projects/BreslowTotal/src/semi_supervision/test/summary/'
create_dir(PATH_OUT)





import argparse
argv = sys.argv[1:]
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-a', '--APPROACH', help='approach to use',type=str, default='Fully_Supervised')
parser.add_argument('-m', '--MODEL', help='model to use',type=str, default='densenet121')
parser.add_argument('-t', '--TASK', help='task to perform',type=str, default='Multiclass')

args = parser.parse_args(argv)
APPROACH = args.APPROACH
MODEL = args.MODEL
TASK = args.TASK


if TASK == 'Multiclass':
    metrics = ['acc_balanced', 'acc', 'f1_score', 'kappa_score', 'precision', 'roc_auc']
    legend = np.array(['Fold', 'acc_balanced', 'acc', 'f1_score', 'kappa_score', 'precision', 'roc_auc_score'], dtype=object)
    reshape_size = (1, 7)
elif TASK == 'InSitu' or TASK == 'Breslow':
    metrics = ['acc_balanced', 'acc', 'f1_score', 'kappa_score', 'precision', 'roc_auc', 'recall', 'specificity']
    legend = np.array(['Fold', 'acc_balanced', 'acc', 'f1_score', 'kappa_score', 'precision', 'roc_auc_score', 'recall', 'specificity'], dtype=object)
    reshape_size = (1, 9)
N_EXPS = ['10', '20', '30']
pd_row = pd.DataFrame()



for N_EXP in N_EXPS:
    try:
        PATH = '/home/jpdominguez/projects/BreslowTotal/src/semi_supervision/train/models/' + MODEL + '/' + TASK + '/N_EXP_' + N_EXP + '/'

        results_per_fold = [[] for i in range(5)]

        pd_row = pd.concat([pd_row, pd.DataFrame(['N_EXP_' + N_EXP])], ignore_index=True)
        pd_row = pd.concat([pd_row, pd.DataFrame(legend.reshape(reshape_size))], ignore_index=True)
        for i in range(5):
            path_local = PATH + 'Fold_' + str(i) + '/checkpoints/'
            N_EXP_local = []
            N_EXP_local.append('Fold_' + str(i))
            for metric in metrics:
                local = pd.read_csv(path_local + 'external_test_' + metric + '.csv', header=None)
                N_EXP_local.append("{:.4f}".format(local[0].values[0]))
                results_per_fold[i].append(local[0].values[0])
                
            N_EXP_local = np.array(N_EXP_local, dtype=object)
            pd_row = pd.concat([pd_row, pd.DataFrame(N_EXP_local.reshape(reshape_size))], ignore_index=True)

        mean_results = []
        std_results = []
        for i in range(len(results_per_fold[0])):
            results_i_per_fold = [results_per_fold[j][i] for j in range(5)]
            results_i_per_fold = np.array(results_i_per_fold)
            mean_results_i = "{:.4f}".format(np.mean(results_i_per_fold))
            mean_results.append(mean_results_i)
            std_results_i = "{:.4f}".format(np.std(results_i_per_fold))
            std_results.append(std_results_i)
        
        
        mean_results = ['Mean'] + mean_results    
        mean_results = np.array(mean_results, dtype = object).reshape(reshape_size)
        std_results = ['Std'] + std_results    
        std_results = np.array(std_results, dtype = object).reshape(reshape_size)
        
        pd_row = pd.concat([pd_row, pd.DataFrame(mean_results)], ignore_index=True)
        pd_row = pd.concat([pd_row, pd.DataFrame(std_results)], ignore_index=True)
        pd_row = pd.concat([pd_row, pd.DataFrame([''])], ignore_index=True)
        pd_row = pd.concat([pd_row, pd.DataFrame([''])], ignore_index=True)
        pd_row = pd.concat([pd_row, pd.DataFrame([''])], ignore_index=True)
    
    except:
        print('N_EXP_' + N_EXP + ' not found')

    
    pd_row.to_excel(PATH_OUT + 'external_' + MODEL + '_' + TASK + '_' + '_'.join(str(x) for x in N_EXPS) + '.xlsx', header=False, index=False)

