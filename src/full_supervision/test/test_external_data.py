import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch.utils.data
from sklearn import metrics 
import os
import shutil
import sys, getopt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse


import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)
torch.manual_seed(0)

argv = sys.argv[1:]

print( torch.cuda.current_device())
print( torch.cuda.device_count())

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)
parser.add_argument('-a', '--APPROACH', help='approach to use',type=str, default='Fully_Supervised')
parser.add_argument('-m', '--MODEL', help='model to use',type=str, default='densenet121')
parser.add_argument('-c', '--NUM_CLASSES', help='number of classes',type=int, default=3)
parser.add_argument('-t', '--TASK', help='task to perform',type=str, default='Multiclass')



args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
APPROACH = args.APPROACH
MODEL = args.MODEL
NUM_CLASSES = args.NUM_CLASSES
TASK = args.TASK

CSV_PATH_EXTERNAL = '/home/jpdominguez/projects/BreslowTotal/data/VdR_new_images_metadata_TOTEST/external_data_paths.csv'

print("TESTING FULLY SUPERVISED MODEL, N_EXP ", N_EXP_str, ', BACKBONE:', MODEL)

def create_dir(directory):
	if not os.path.isdir(directory):
		try:
			os.mkdir(directory)
		except OSError:
			print ("Creation of the directory %s failed" % directory)
		else:
			print ("Successfully created the directory %s " % directory)


def plot_confusion_matrix(cm,
                          target_names,
                          title='AOEC patch predictions',
                          cmap=None,
                          normalize=True,
                         metric = None,
						 path_to_save = None):

	import matplotlib.pyplot as plt
	import numpy as np
	import itertools
    
	import matplotlib.pyplot as plt
	import numpy as np
	import itertools
	import pylab

	Fi = pylab.gcf()
	DefaultSize = Fi.get_size_inches()

	fig = plt.gcf()
	DPI = fig.get_dpi()
	fig.set_size_inches(1800.0/float(DPI),1800.0/float(DPI))
    
	dpi = 600
	fontsize = 36
    
	kappa = metric #0.6613

	if cmap is None:
		cmap = plt.get_cmap('Blues')
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
        
	plt.figure(figsize=(16, 12))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title, fontsize=25)
    #plt.colorbar()
    
	cb = plt.colorbar()
	for t in cb.ax.get_yticklabels():
		t.set_fontsize(fontsize)
    
	if target_names is not None:
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation=0,fontsize=fontsize, ha='center')
		plt.yticks(tick_marks, target_names,rotation=90, fontsize=fontsize, va='center')

    
	thresh = cm.max() / 1.5 if normalize else cm.max() / 2

	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if normalize:
			plt.text(j, i, "{:0.3f}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black",fontsize=fontsize)
		else:
			plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=fontsize)

	plt.ylabel('Ground truth',fontsize=fontsize)
	plt.xlabel('Model predictions (AUC = {:0.4f})'.format(kappa),fontsize=fontsize)
	plt.tight_layout()
	plt.savefig(path_to_save, format='svg',dpi=dpi)


def	plot_roc_curve(fpr,
					tpr,
					xlabel = 'False positive rate',
					ylabel = 'True positive rate',
					title        = "ROC curve",
					metric = None,
					path_to_save = None):
	lw = 4
	# plt.figure()
	
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('Receiver operating characteristic example')
	# plt.legend(loc="lower right")
	# plt.savefig(OUTPUT_PATH_FOLD+'ROC_curve.png')
	# plt.close()


	import matplotlib.pyplot as plt
	import numpy as np
	import itertools
    
	import matplotlib.pyplot as plt
	import numpy as np
	import itertools
	import pylab

	Fi = pylab.gcf()
	DefaultSize = Fi.get_size_inches()

	fig = plt.gcf()
	DPI = fig.get_dpi()
	fig.set_size_inches(1800.0/float(DPI),1800.0/float(DPI))
    
	dpi = 600
	fontsize = 40    
        
	plt.figure(figsize=(16, 12))
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.plot([0, 1], [0, 1], 'k-', lw=lw/2)
	plt.plot(fpr, tpr, lw=lw, label='ROC curve (area = {:0.4f})'.format(metric))
	plt.xlabel('False Positive Rate', fontsize=fontsize)
	plt.ylabel('True Positive Rate', fontsize=fontsize)	
	plt.legend(loc="lower right", fontsize=fontsize)
	plt.tight_layout()
	plt.savefig(path_to_save, format='svg',dpi=dpi)


def specificity(y_true, y_pred): 
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity

imageNet_weights = True


class StudentModel(torch.nn.Module):
	def __init__(self):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(StudentModel, self).__init__()

		if MODEL == 'vgg16':
			pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', 'vgg16', pretrained=imageNet_weights)
			print('Loading ImageNet weights')
			fc_input_features = 512
		elif MODEL == 'densenet121':
			pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', 'densenet121', pretrained=imageNet_weights)
			print('Loading ImageNet weights')
			fc_input_features = pre_trained_network.classifier.in_features
		elif MODEL == 'inception_v3':
			pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', 'inception_v3', pretrained=imageNet_weights)
			print('Loading ImageNet weights')
			fc_input_features = pre_trained_network.fc.in_features
		elif MODEL == 'resnet50':
			pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=imageNet_weights)
			print('Loading ImageNet weights')
			fc_input_features = pre_trained_network.fc.in_features

		self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])

		if (torch.cuda.device_count()>1):
			self.conv_layers = torch.nn.DataParallel(self.conv_layers)

		
		self.fc_feat_in = fc_input_features
		self.N_CLASSES = NUM_CLASSES

		self.E = 128

		self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
		self.fc = torch.nn.Linear(in_features=self.E, out_features=self.N_CLASSES)
		self.relu = torch.nn.ReLU()

	def forward(self, x, conv_layers_out):
			"""
			In the forward function we accept a Tensor of input data and we must return
			a Tensor of output data. We can use Modules defined in the constructor as
			well as arbitrary operators on Tensors.
			"""
			A = None
			m_binary = torch.nn.Sigmoid()
			m_multiclass = torch.nn.Softmax()

			dropout = torch.nn.Dropout(p=0.2)
			
			if x is not None:
				conv_layers_out=self.conv_layers(x)
				
				n = torch.nn.AdaptiveAvgPool2d((1,1))
				conv_layers_out = n(conv_layers_out)
				
				conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)

			embedding_layer = self.embedding(conv_layers_out)
			embedding_layer = self.relu(embedding_layer)
			features_to_return = embedding_layer
			embedding_layer = dropout(embedding_layer)

			logits = self.fc(embedding_layer)

			output_fcn = m_multiclass(logits)
			
			return logits, output_fcn


models_path = '/home/jpdominguez/projects/BreslowTotal/src/full_supervision/train/models/'
create_dir(models_path)
models_path = models_path+MODEL+'/'
create_dir(models_path)
models_path = models_path+TASK+'/'
create_dir(models_path)
models_path = models_path+'N_EXP_'+N_EXP_str+'/'
create_dir(models_path)


from torchvision import transforms
#DATA AUGMENTATION

#DATA NORMALIZATION
preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	transforms.Resize((224,224)),
])

class Dataset_test(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.list_IDs = list_IDs
		self.labels = labels
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):

		# Select sample
		ID = self.list_IDs[index]
		#print(ID)
		# Load data and get label
		X = Image.open(ID)
		X = X.convert('RGB')
		X = np.asarray(X)
		y = self.labels[index]

		#data transformation
		input_tensor = preprocess(X)
				
		return input_tensor, np.asarray(y)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
params_test = {'batch_size': BATCH_SIZE,
		  'shuffle': False,
		  #'sampler': ImbalancedDatasetSampler(test_dataset),
		  'num_workers': 0}


# DATA LOADING

csv_df_vdr = pd.read_csv(CSV_PATH_EXTERNAL, sep=";", encoding='unicode_escape') 

class_vdr = []
for i in range(len(csv_df_vdr)):
	if csv_df_vdr.iloc[i]['BRESLOW'] == '1,66-2,42':
		class_vdr.append(2)
	# elif csv_df_vdr.iloc[i]['BRESLOW'] != csv_df_vdr.iloc[i]['BRESLOW']: # check if Breslow value is NaN
	# 	print('HEY')
	# 	csv_df_vdr = csv_df_vdr.drop(index=i)
	# 	#csv_df_vdr.reset_index()
	# 	#i=i-1
	# 	ending_index = ending_index - 1
	# 	continue
	elif csv_df_vdr.iloc[i]['BRESLOW'] == 'Tis' or float(csv_df_vdr.iloc[i]['BRESLOW'].replace(',', '.')) == 0:
		class_vdr.append(0)
	elif float(csv_df_vdr.iloc[i]['BRESLOW'].replace(',', '.')) < 0.8 and float(csv_df_vdr.iloc[i]['BRESLOW'].replace(',', '.')) != 0:
		class_vdr.append(1)
	elif float(csv_df_vdr.iloc[i]['BRESLOW'].replace(',', '.')) >=  0.8:
		class_vdr.append(2)
	else:
		print(csv_df_vdr.iloc[i]['BRESLOW'])
csv_df_vdr['CLASS'] = class_vdr


if NUM_CLASSES == 2:
	if TASK == 'Breslow':
		csv_df_vdr = csv_df_vdr[csv_df_vdr['CLASS'] != 0]
		csv_df_vdr['CLASS'] = [0 if x == 1 else 1 for x in csv_df_vdr['CLASS']]
	elif TASK == 'InSitu':
		csv_df_vdr['CLASS'] = [0 if x == 0 else 1 for x in csv_df_vdr['CLASS'] ]

csv_df_vdr_shuffled = csv_df_vdr.sample(frac = 1, random_state= np.random.RandomState()).reset_index(drop=True)
csv_df_vdr_shuffled_iloc = csv_df_vdr_shuffled #.iloc[:][:]

print('------------------------')
print('Virgen del Rocío')
print('Images found:', len(csv_df_vdr_shuffled), '--', np.all([os.path.isfile(i) for i in csv_df_vdr_shuffled_iloc['IMAGE_PATH']]))
if NUM_CLASSES == 3:
	print('In situ:', len(csv_df_vdr_shuffled[csv_df_vdr_shuffled.CLASS == 0]))
	print('Breslow < 0.8:', len(csv_df_vdr_shuffled[csv_df_vdr_shuffled.CLASS == 1]))
	print('Breslow >= 0.8:', len(csv_df_vdr_shuffled[csv_df_vdr_shuffled.CLASS == 2]))
elif NUM_CLASSES == 2 and TASK == 'InSitu':
	print('In situ:', len(csv_df_vdr_shuffled[csv_df_vdr_shuffled.CLASS == 0]))
	print('Invasive:', len(csv_df_vdr_shuffled[csv_df_vdr_shuffled.CLASS == 1]))
elif NUM_CLASSES == 2 and TASK == 'Breslow':
	print('Breslow < 0.8:', len(csv_df_vdr_shuffled[csv_df_vdr_shuffled.CLASS == 0]))
	print('Breslow >= 0.8:', len(csv_df_vdr_shuffled[csv_df_vdr_shuffled.CLASS == 1]))
print('------------------------')



for i in range(5):
	print("Running Fold", i+1, "/", 5)
	OUTPUT_PATH_FOLD = models_path + 'Fold_'+str(i) + os.sep
	model_path = OUTPUT_PATH_FOLD+'fully_supervised_model_strongly.pt'

	model = torch.load(model_path)
	# test_dataset = pd.read_csv(OUTPUT_PATH_FOLD + 'validation_df.csv')#,header=None)
	# test_dataset = test_dataset.values
	test_dataset = csv_df_vdr_shuffled_iloc.values

	checkpoint_path = OUTPUT_PATH_FOLD+'checkpoints/'
	create_dir(checkpoint_path)	


	model = torch.load(model_path)
	model.eval()

	testing_set = Dataset_test(test_dataset[:,0], test_dataset[:,-1])
	testing_generator = data.DataLoader(testing_set, **params_test)

	y_pred = []
	y_true = []
	output_preds_0 = []
	output_preds_1 = []
	output_preds_2 = []
	outputs_np_winner = []

	with torch.no_grad():
		j = 0
		for inputs,labels in testing_generator:
			inputs, labels = inputs.to(device), labels.to(device)

			logits, outputs = model(inputs, None)

			outputs_np = outputs.cpu().data.numpy()
			labels_np = labels.cpu().data.numpy()
			outputs_np = np.argmax(outputs_np, axis=1)
			y_pred = np.append(y_pred,outputs_np)
			y_true = np.append(y_true,labels_np)

			outputs_np = outputs.cpu().data.numpy()
			labels_np = labels.cpu().data.numpy()
			output_preds_0 = np.append(output_preds_0, outputs_np[:, [0]])
			output_preds_1 = np.append(output_preds_1, outputs_np[:, [1]])

			# print(outputs_np)
			# outputs_np_winner = np.argmax(outputs_np, axis=1)

			if outputs_np_winner == []:
				outputs_np_winner = outputs_np
			else:
				outputs_np_winner = np.concatenate((outputs_np_winner, outputs_np), axis=0)

			if NUM_CLASSES == 3:
				output_preds_2 = np.append(output_preds_2, outputs_np[:, [2]])


	# if NUM_CLASSES == 2:
	# 	# plot ROC curve
	# 	print(len(y_true), len(outputs_np_winner))
	# 	fpr, tpr, thresholds = metrics.roc_curve(y_true, outputs_np_winner)
	# 	roc_auc = metrics.auc(fpr, tpr)
	# 	plt.figure()
	# 	lw = 2
	# 	plt.plot(fpr, tpr, color='darkorange',
	# 	  				 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)


	#k-score
	k_score = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
	print("k_score " + str(k_score))
	#f1_score
	f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
	print("f1_score " + str(f1_score))
	#acc
	acc = metrics.accuracy_score(y_true, y_pred)
	print("acc " + str(acc))
	#precision
	precision = metrics.precision_score(y_true, y_pred, average='weighted')
	print("precision " + str(precision))

	# #recall
	# recall = metrics.recall_score(y_true, y_pred, average='weighted')
	# print("recall " + str(recall))
	# if NUM_CLASSES == 2:
	# 	#specificity
	# 	# specificity_score = specificity(y_true, y_pred)
	# 	specificity_score = metrics.recall_score(y_true, y_pred, pos_label=0, average='weighted')
	# 	print("specificity " + str(specificity_score))
	# 	#roc_auc
	# 	roc_auc_score = metrics.roc_auc_score(y_true, y_pred)
	# 	print("roc_auc " + str(roc_auc_score))

	if NUM_CLASSES == 2:
		#recall
		recall = metrics.recall_score(y_true, y_pred)
		print("recall " + str(recall))
		#specificity
		specificity_score = metrics.recall_score(y_true, y_pred, pos_label=0)
		print("specificity " + str(specificity_score))
	
	#roc_auc
	if NUM_CLASSES == 2:
		# outputs_np_winner = np.max(outputs_np_winner, axis=1)
		# print(outputs_np_winner)
		roc_auc_score = metrics.roc_auc_score(y_true, output_preds_1)
	elif NUM_CLASSES == 3:		
		roc_auc_score = metrics.roc_auc_score(y_true, outputs_np_winner, multi_class='ovr', average='weighted')
	
	print("roc_auc " + str(roc_auc_score))
	#confusion matrix
	confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
	print("confusion_matrix ")
	print(str(confusion_matrix))
	#acc_balanced
	acc_balanced = metrics.balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
	print("acc_balanced " + str(acc_balanced))


	kappa_score_general_filename = checkpoint_path+'external_test_kappa_score.csv'
	acc_balanced_filename = checkpoint_path+'external_test_acc_balanced.csv'
	confusion_matrix_filename = checkpoint_path+'external_test_conf_matr.csv'
	roc_auc_score_filename = checkpoint_path+'external_test_roc_auc.csv'
	f1_score_filename = checkpoint_path+'external_test_f1_score.csv'
	accc_filename = checkpoint_path+'external_test_acc.csv'
	precision_filename = checkpoint_path+'external_test_precision.csv'
	recall_filename = checkpoint_path+'external_test_recall.csv'
	specificity_filename = checkpoint_path+'external_test_specificity.csv'


	kappas = [k_score]

	File = {'val':kappas}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(kappa_score_general_filename, df.values, fmt='%s',delimiter=',')

	f1_scores = [f1_score]

	File = {'val':f1_scores}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(f1_score_filename, df.values, fmt='%s',delimiter=',')

	acc_balancs = [acc_balanced]

	File = {'val':acc_balancs}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(acc_balanced_filename, df.values, fmt='%s',delimiter=',')

	conf_matr = [confusion_matrix]
	File = {'val':conf_matr}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(confusion_matrix_filename, df.values, fmt='%s',delimiter=',')

	preds_filename = checkpoint_path + 'preds_raw.csv' #checkpoint_path+'kappa_score_general_'+DATASET+'_strong.csv'

	if NUM_CLASSES == 2:
		File = {'y_gt':y_true,'y_pred_0':output_preds_0,'y_pred_1':output_preds_1,}
		df = pd.DataFrame(File,columns=['y_gt','y_pred_0','y_pred_1'])
		np.savetxt(preds_filename, df.values, fmt='%.14f',delimiter=',')
	elif NUM_CLASSES == 3:
		File = {'y_gt':y_true,'y_pred_0':output_preds_0,'y_pred_1':output_preds_1,'y_pred_2':output_preds_2}
		df = pd.DataFrame(File,columns=['y_gt','y_pred_0','y_pred_1','y_pred_2'])
		np.savetxt(preds_filename, df.values, fmt='%.14f',delimiter=',')

	accs = [acc]

	File = {'val':accs}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(accc_filename, df.values, fmt='%s',delimiter=',')

	precisions = [precision]

	File = {'val':precisions}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(precision_filename, df.values, fmt='%s',delimiter=',')

	roc_auc = [roc_auc_score]
	File = {'val':roc_auc}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(roc_auc_score_filename, df.values, fmt='%s',delimiter=',')



	if NUM_CLASSES == 2:
		recalls = [recall]

		File = {'val':recalls}
		df = pd.DataFrame(File,columns=['val'])
		np.savetxt(recall_filename, df.values, fmt='%s',delimiter=',')


		specificitys = [specificity_score]

		File = {'val':specificitys}
		df = pd.DataFrame(File,columns=['val'])
		np.savetxt(specificity_filename, df.values, fmt='%s',delimiter=',')


		fpr, tpr, _ = metrics.roc_curve(y_true, output_preds_1)
		plot_roc_curve(fpr = fpr,
						tpr = tpr,
						xlabel = 'False positive rate',
						ylabel = 'True positive rate',
						title        = "",
						metric       = np.array(roc_auc_score),
						path_to_save=checkpoint_path + 'external_test_plot_roc_curve.svg')

	cm = np.array(confusion_matrix)
	target_names = ['In situ', 'Breslow\n< 0.8 mm', 'Breslow\n>= 0.8 mm']
	if NUM_CLASSES == 2 and TASK == 'Breslow':
		target_names = ['Breslow < 0.8 mm', 'Breslow >= 0.8 mm']
	elif NUM_CLASSES == 2 and TASK == 'InSitu':
		target_names = ['In situ', 'Invasive']
	plot_confusion_matrix(cm           = cm, 
						normalize    = True,
						target_names = target_names,
						title        = "",
						# metric       = np.array(roc_auc_score),
						metric = np.array(roc_auc_score),
						path_to_save=checkpoint_path + 'external_plot_cm_breslow.svg')
