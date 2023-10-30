import numpy as np
import pandas as pd
import os
import shutil

import argparse


import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)


#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-d', '--DATASET', help='dataset to test',type=str, default='all')
parser.add_argument('-m', '--MODEL', help='model to use',type=str, default='densenet121')


args = parser.parse_args()

DATASET = args.DATASET
MODEL = args.MODEL
N_EXP = args.N_EXP

PATH_RESULTS = 'E:\\Breslow\\src\\pytorch\\experiments\\InSitu_Invassive\\semi_supervision\\'
MODELS_RANGE = range(0,5)

dir_results = PATH_RESULTS + 'train\\models\\' + '\\' + MODEL + '\\'


def create_dir(directory):
	if not os.path.isdir(directory):
		try:
			os.mkdir(directory)
		except OSError:
			print ("Creation of the directory %s failed" % directory)
		else:
			print ("Successfully created the directory %s " % directory)

dir_output = PATH_RESULTS + '\\test\\mean_and_best\\'
create_dir(dir_output)
dir_output = dir_output + MODEL + '\\'
create_dir(dir_output)
dir_output = dir_output + 'N_EXP_' + str(N_EXP) + '\\'
create_dir(dir_output)



kappa_score = []
f1_score = []
acc_balanced = []
acc = []
precision = []
recall = []
specificity = []
auc = []

for i in MODELS_RANGE:
    dir_N_EXP = dir_results + 'N_EXP_' + str(N_EXP) + '\\Fold_' + str(i) + '\\checkpoints\\'

    k = pd.read_csv(dir_N_EXP + 'kappa_score_general_strong.csv').columns[0]
    f = pd.read_csv(dir_N_EXP + 'f1_score_general_strong.csv').columns[0]
    ab = pd.read_csv(dir_N_EXP + 'acc_balanced_general_strong.csv').columns[0]
    ac = pd.read_csv(dir_N_EXP + 'acc_general_strong.csv').columns[0]
    p = pd.read_csv(dir_N_EXP + 'precision_general_strong.csv').columns[0]
    r = pd.read_csv(dir_N_EXP + 'recall_general_strong.csv').columns[0]
    s = pd.read_csv(dir_N_EXP + 'specificity_general_strong.csv').columns[0]
    au = pd.read_csv(dir_N_EXP + 'roc_auc_score_general_strong.csv').columns[0]

    # print('kappa_score')
    print(au)
    # print('f1_score')
    # print(f)
    # print('acc_balanced')
    # print(a)

    kappa_score.append(float(k))
    f1_score.append(float(f))
    acc_balanced.append(float(ab))
    acc.append(float(ac))
    precision.append(float(p))
    recall.append(float(r))
    specificity.append(float(s))
    auc.append(float(au))


print('kappa_score', np.mean(kappa_score), '+-', np.std(kappa_score))
print('f1_score', np.mean(f1_score), '+-', np.std(f1_score))
print('acc_balanced', np.mean(acc_balanced), '+-', np.std(acc_balanced))
print('acc', np.mean(acc), '+-', np.std(acc))
print('precision', np.mean(precision), '+-', np.std(precision))
print('recall', np.mean(recall), '+-', np.std(recall))
print('specificity', np.mean(specificity), '+-', np.std(specificity))
print('auc', np.mean(auc), '+-', np.std(auc))


#copy best model
best_index = np.argmax(auc)
dir_best_N_EXP = dir_results + 'N_EXP_' + str(N_EXP) + '\\Fold_' + str(best_index) + '\\checkpoints\\'
# copy file to new directory
shutil.copy(dir_best_N_EXP + 'kappa_score_general_strong.csv', dir_output + 'best_kappa_score_general_strong.csv')
shutil.copy(dir_best_N_EXP + 'f1_score_general_strong.csv', dir_output + 'best_f1_score_general_strong.csv')
shutil.copy(dir_best_N_EXP + 'acc_balanced_general_strong.csv', dir_output + 'best_acc_balanced_general_strong.csv')
shutil.copy(dir_best_N_EXP + 'acc_general_strong.csv', dir_output + 'best_acc_general_strong.csv')
shutil.copy(dir_best_N_EXP + 'precision_general_strong.csv', dir_output + 'best_precision_general_strong.csv')
shutil.copy(dir_best_N_EXP + 'recall_general_strong.csv', dir_output + 'best_recall_general_strong.csv')
shutil.copy(dir_best_N_EXP + 'specificity_general_strong.csv', dir_output + 'best_specificity_general_strong.csv')
shutil.copy(dir_best_N_EXP + 'roc_auc_score_general_strong.csv', dir_output + 'best_roc_auc_score_general_strong.csv')
shutil.copy(dir_best_N_EXP + 'plot_cm_InSitu_Invassive.svg', dir_output + 'best_plot_cm_InSitu_Invassive.svg')
shutil.copy(dir_best_N_EXP + 'plot_roc_curve_InSitu_Invassive.svg', dir_output + 'best_plot_roc_curve_InSitu_Invassive.svg')

# save mean and std metrics in csv
df = pd.DataFrame([np.mean(kappa_score),np.std(kappa_score)])
df.to_csv(dir_output + 'mean_and_std_kappa_score_strong.csv', index=False, header=False)
df = pd.DataFrame([np.mean(f1_score),np.std(f1_score)])
df.to_csv(dir_output + 'mean_and_std_f1_score_strong.csv', index=False, header=False)
df = pd.DataFrame([np.mean(acc_balanced),np.std(acc_balanced)])
df.to_csv(dir_output + 'mean_and_std_acc_balanced_strong.csv', index=False, header=False)
df = pd.DataFrame([np.mean(acc),np.std(acc)])
df.to_csv(dir_output + 'mean_and_std_acc_strong.csv', index=False, header=False)
df = pd.DataFrame([np.mean(precision),np.std(precision)])
df.to_csv(dir_output + 'mean_and_std_precision_strong.csv', index=False, header=False)
df = pd.DataFrame([np.mean(recall),np.std(recall)])
df.to_csv(dir_output + 'mean_and_std_recall_strong.csv', index=False, header=False)
df = pd.DataFrame([np.mean(specificity),np.std(specificity)])
df.to_csv(dir_output + 'mean_and_std_specificity_strong.csv', index=False, header=False)
df = pd.DataFrame([np.mean(auc),np.std(auc)])
df.to_csv(dir_output + 'mean_and_std_roc_auc_score_strong.csv', index=False, header=False)
