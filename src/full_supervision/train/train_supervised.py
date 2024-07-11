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
import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

np.random.seed(0)

argv = sys.argv[1:]


import warnings
warnings.filterwarnings('ignore')


#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)
parser.add_argument('-e', '--EPOCHS', help='number of epochs',type=int, default=15)
parser.add_argument('-m', '--MODEL', help='model to use',type=str, default='densenet121')
parser.add_argument('-l', '--LEARNING_RATE', help='learning rate to use',type=float, default=0.00001)
parser.add_argument('-c', '--NUM_CLASSES', help='number of classes',type=int, default=3)
parser.add_argument('-t', '--TASK', help='task to perform',type=str, default='Multiclass')


args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
EPOCHS = args.EPOCHS
MODEL = args.MODEL
LEARNING_RATE = args.LEARNING_RATE
N_CLASSES = args.NUM_CLASSES
TASK = args.TASK


PARALLEL = False
DEV_GPU = 0

print("TRAINING FULLY SUPERVISED NETWORK, N_EXP " + N_EXP_str, "BATCH_SIZE " + BATCH_SIZE_str, "EPOCHS " + str(EPOCHS), "MODEL " + MODEL, "LEARNING_RATE " + str(LEARNING_RATE), "N_CLASSES " + str(N_CLASSES), "TASK " + TASK)

def create_dir(directory):
	if not os.path.isdir(directory):
		try:
			os.mkdir(directory)
		except OSError:
			print ("Creation of the directory %s failed" % directory)
		else:
			print ("Successfully created the directory %s " % directory) 


models_path = '/home/jpdominguez/projects/BreslowTotal/src/full_supervision/train/models/'
create_dir(models_path)
models_path = models_path+MODEL+'/'
create_dir(models_path)
models_path = models_path+TASK+'/'
create_dir(models_path)
models_path = models_path+'N_EXP_'+N_EXP_str+'/'
create_dir(models_path) 


CSV_PATH_VDR = '/home/jpdominguez/projects/BreslowTotal/data/melanomas virgen del rocio/Total de imagenes con Breslow-VR/images_paths_whole.csv'
CSV_PATH_ARG = '/home/jpdominguez/projects/BreslowTotal/data/Argenciano/images_paths_whole.csv'
CSV_PATH_ISIC = '/home/jpdominguez/projects/BreslowTotal/data/ISIC con Breslow 27-01-23/images3/isic_test.csv'
CSV_PATH_POLESIE = '/home/jpdominguez/projects/BreslowTotal/data/Imagenes Polesie et al 2021/images_paths_whole.csv'


# VIRGEN DEL ROCIO

csv_df_vdr = pd.read_csv(CSV_PATH_VDR) 

class_vdr = []
for i in range(len(csv_df_vdr)):
	if csv_df_vdr.iloc[i]['BRESLOW'] == 'Tis':
		class_vdr.append(0)
	elif float(csv_df_vdr.iloc[i]['BRESLOW'].replace(',', '.')) < 0.8 and float(csv_df_vdr.iloc[i]['BRESLOW'].replace(',', '.')) != 0:
		class_vdr.append(1)
	elif float(csv_df_vdr.iloc[i]['BRESLOW'].replace(',', '.')) >=  0.8:
		class_vdr.append(2)
csv_df_vdr['CLASS'] = class_vdr

if N_CLASSES == 2:
	if TASK == 'Breslow':
		csv_df_vdr = csv_df_vdr[csv_df_vdr['CLASS'] != 0]
		csv_df_vdr['CLASS'] = [0 if x == 1 else 1 for x in csv_df_vdr['CLASS']]
	elif TASK == 'InSitu':
		csv_df_vdr['CLASS'] = [0 if x == 0 else 1 for x in csv_df_vdr['CLASS'] ]

csv_df_vdr_shuffled = csv_df_vdr.sample(frac = 1, random_state= np.random.RandomState()).reset_index()
csv_df_vdr_shuffled_iloc = csv_df_vdr_shuffled.iloc[:][:]

print('------------------------')
print('Virgen del Rocío')
print('Images found:', len(csv_df_vdr_shuffled), '--', np.all([os.path.isfile(i) for i in csv_df_vdr_shuffled_iloc['IMAGE_PATH']]))
print('In situ:', len(csv_df_vdr_shuffled[csv_df_vdr_shuffled.CLASS == 0]))
print('Breslow < 0.8:', len(csv_df_vdr_shuffled[csv_df_vdr_shuffled.CLASS == 1]))
print('Breslow >= 0.8:', len(csv_df_vdr_shuffled[csv_df_vdr_shuffled.CLASS == 2]))
print('------------------------')


# ARGENZIANO

csv_df_arg = pd.read_csv(CSV_PATH_ARG) 
class_arg = []
csv_df_arg = csv_df_arg[csv_df_arg.BRESLOW_processed != '-'].reset_index()
for i in range(len(csv_df_arg)):
	if csv_df_arg.iloc[i]['BRESLOW_processed'] == 'X' and csv_df_arg.iloc[i]['IN_SITU'] == 1:
		class_arg.append(0)
	elif int(csv_df_arg.iloc[i]['BRESLOW_processed'].replace(',', '.')) == 0:
		class_arg.append(1)
	elif int(csv_df_arg.iloc[i]['BRESLOW_processed'].replace(',', '.')) == 1:
		class_arg.append(2)
csv_df_arg['CLASS'] = class_arg

if N_CLASSES == 2:
	if TASK == 'Breslow':
		csv_df_arg = csv_df_arg[csv_df_arg['CLASS'] != 0]
		csv_df_arg['CLASS'] = [0 if x == 1 else 1 for x in csv_df_arg['CLASS']]
	elif TASK == 'InSitu':
		csv_df_arg['CLASS'] = [0 if x == 0 else 1 for x in csv_df_arg['CLASS'] ]

csv_df_arg_shuffled = csv_df_arg.sample(frac = 1, random_state= np.random.RandomState()).reset_index()
csv_df_arg_shuffled_iloc = csv_df_arg_shuffled.iloc[:][:]

print('------------------------')
print('Argenziano')
print('Images found:', len(csv_df_arg_shuffled), '--', np.all([os.path.isfile(i) for i in csv_df_arg_shuffled_iloc['IMAGE_PATH']]))
print('In situ:', len(csv_df_arg_shuffled[csv_df_arg_shuffled.CLASS == 0]))
print('Breslow < 0.8:', len(csv_df_arg_shuffled[csv_df_arg_shuffled.CLASS == 1]))
print('Breslow >= 0.8:', len(csv_df_arg_shuffled[csv_df_arg_shuffled.CLASS == 2]))
print('------------------------')


# POLESIE

csv_df_polesie = pd.read_csv(CSV_PATH_POLESIE) 
class_polesie = []
csv_df_polesie = csv_df_polesie[csv_df_polesie.BRESLOW_processed != '-'].reset_index()
for i in range(len(csv_df_polesie)):
	if csv_df_polesie.iloc[i]['BRESLOW_processed'] == 'X' and csv_df_polesie.iloc[i]['IN_SITU'] == 1:
		class_polesie.append(0)
	elif int(csv_df_polesie.iloc[i]['BRESLOW_processed'].replace(',', '.')) == 0:
		class_polesie.append(1)
	elif int(csv_df_polesie.iloc[i]['BRESLOW_processed'].replace(',', '.')) == 1:
		class_polesie.append(2)
csv_df_polesie['CLASS'] = class_polesie

if N_CLASSES == 2:
	if TASK == 'Breslow':
		csv_df_polesie = csv_df_polesie[csv_df_polesie['CLASS'] != 0]
		csv_df_polesie['CLASS'] = [0 if x == 1 else 1 for x in csv_df_polesie['CLASS']]
	elif TASK == 'InSitu':
		csv_df_polesie['CLASS'] = [0 if x == 0 else 1 for x in csv_df_polesie['CLASS'] ]

csv_df_polesie_shuffled = csv_df_polesie.sample(frac = 1, random_state= np.random.RandomState()).reset_index()
csv_df_polesie_shuffled_iloc = csv_df_polesie_shuffled.iloc[:][:]

print('------------------------')
print('Polesie')
print('Images found:', len(csv_df_polesie_shuffled), '--', np.all([os.path.isfile(i) for i in csv_df_polesie_shuffled_iloc['IMAGE_PATH']]))
print('In situ:', len(csv_df_polesie_shuffled[csv_df_polesie_shuffled.CLASS == 0]))
print('Breslow < 0.8:', len(csv_df_polesie_shuffled[csv_df_polesie_shuffled.CLASS == 1]))
print('Breslow >= 0.8:', len(csv_df_polesie_shuffled[csv_df_polesie_shuffled.CLASS == 2]))
print('------------------------')


# ISIC

csv_df_isic = pd.read_csv(CSV_PATH_ISIC)
class_isic = []
for i in range(len(csv_df_isic)):
	if float(csv_df_isic.iloc[i]['mel_thick_mm']) == 0.0:
		class_isic.append(0)
	elif float(csv_df_isic.iloc[i]['mel_thick_mm']) < 0.8 and float(csv_df_isic.iloc[i]['mel_thick_mm']):
		class_isic.append(1)
	elif float(csv_df_isic.iloc[i]['mel_thick_mm']) >= 0.8:
		class_isic.append(2)
csv_df_isic['CLASS'] = class_isic

if N_CLASSES == 2:
	if TASK == 'Breslow':
		csv_df_isic = csv_df_isic[csv_df_isic['CLASS'] != 0]
		csv_df_isic['CLASS'] = [0 if x == 1 else 1 for x in csv_df_isic['CLASS']]
	elif TASK == 'InSitu':
		csv_df_isic['CLASS'] = [0 if x == 0 else 1 for x in csv_df_isic['CLASS'] ]

csv_df_isic_shuffled = csv_df_isic.sample(frac = 1, random_state= np.random.RandomState()).reset_index()
csv_df_isic_shuffled_iloc = csv_df_isic_shuffled.iloc[:][:]
print('------------------------')
print('ISIC')
print('Images found:', len(csv_df_isic_shuffled), '--', np.all([os.path.isfile(i) for i in csv_df_isic_shuffled['IMAGE_PATH']]))
print('In situ:', len(csv_df_isic_shuffled[csv_df_isic_shuffled.CLASS == 0]))
print('Breslow < 0.8:', len(csv_df_isic_shuffled[csv_df_isic_shuffled.CLASS == 1]))
print('Breslow >= 0.8:', len(csv_df_isic_shuffled[csv_df_isic_shuffled.CLASS == 2]))
print('------------------------')



csv_df_vdr_shuffled['CLASS'] = [str(x) for x in csv_df_vdr_shuffled['CLASS']]
csv_df_arg_shuffled['CLASS'] = [str(x) for x in csv_df_arg_shuffled['CLASS']]
csv_df_polesie_shuffled['CLASS'] = [str(x) for x in csv_df_polesie_shuffled['CLASS']]
csv_df_isic_shuffled['CLASS'] = [str(x) for x in csv_df_isic_shuffled['CLASS']]


patients_vdr = np.unique(csv_df_vdr_shuffled_iloc['NHC'])
patients_arg = np.unique(csv_df_arg_shuffled_iloc['ID'])
patients_polesie = np.unique(csv_df_polesie_shuffled_iloc['ID'])
patients_isic = np.unique(csv_df_isic_shuffled_iloc['isic_id'])

np.random.shuffle(patients_vdr)
np.random.shuffle(patients_arg)
np.random.shuffle(patients_polesie)
np.random.shuffle(patients_isic)


fold_1_vdr = patients_vdr[:int(len(patients_vdr)*0.2)]
fold_2_vdr = patients_vdr[int(len(patients_vdr)*0.2):int(len(patients_vdr)*0.4)]
fold_3_vdr = patients_vdr[int(len(patients_vdr)*0.4):int(len(patients_vdr)*0.6)]
fold_4_vdr = patients_vdr[int(len(patients_vdr)*0.6):int(len(patients_vdr)*0.8)]
fold_5_vdr = patients_vdr[int(len(patients_vdr)*0.8):]

fold_1_arg = patients_arg[:int(len(patients_arg)*0.2)]
fold_2_arg = patients_arg[int(len(patients_arg)*0.2):int(len(patients_arg)*0.4)]
fold_3_arg = patients_arg[int(len(patients_arg)*0.4):int(len(patients_arg)*0.6)]
fold_4_arg = patients_arg[int(len(patients_arg)*0.6):int(len(patients_arg)*0.8)]
fold_5_arg = patients_arg[int(len(patients_arg)*0.8):]

fold_1_polesie = patients_polesie[:int(len(patients_polesie)*0.2)]
fold_2_polesie = patients_polesie[int(len(patients_polesie)*0.2):int(len(patients_polesie)*0.4)]
fold_3_polesie = patients_polesie[int(len(patients_polesie)*0.4):int(len(patients_polesie)*0.6)]
fold_4_polesie = patients_polesie[int(len(patients_polesie)*0.6):int(len(patients_polesie)*0.8)]
fold_5_polesie = patients_polesie[int(len(patients_polesie)*0.8):]

fold_1_isic = patients_isic[:int(len(patients_isic)*0.2)]
fold_2_isic = patients_isic[int(len(patients_isic)*0.2):int(len(patients_isic)*0.4)]
fold_3_isic = patients_isic[int(len(patients_isic)*0.4):int(len(patients_isic)*0.6)]
fold_4_isic = patients_isic[int(len(patients_isic)*0.6):int(len(patients_isic)*0.8)]
fold_5_isic = patients_isic[int(len(patients_isic)*0.8):]

fold_1_vdr_df = csv_df_vdr_shuffled[csv_df_vdr_shuffled['NHC'].isin(fold_1_vdr)]#.reset_index()
fold_2_vdr_df = csv_df_vdr_shuffled[csv_df_vdr_shuffled['NHC'].isin(fold_2_vdr)]#.reset_index()
fold_3_vdr_df = csv_df_vdr_shuffled[csv_df_vdr_shuffled['NHC'].isin(fold_3_vdr)]#.reset_index()
fold_4_vdr_df = csv_df_vdr_shuffled[csv_df_vdr_shuffled['NHC'].isin(fold_4_vdr)]#.reset_index()
fold_5_vdr_df = csv_df_vdr_shuffled[csv_df_vdr_shuffled['NHC'].isin(fold_5_vdr)]#.reset_index()

fold_1_arg_df = csv_df_arg_shuffled[csv_df_arg_shuffled['ID'].isin(fold_1_arg)]#.reset_index()
fold_2_arg_df = csv_df_arg_shuffled[csv_df_arg_shuffled['ID'].isin(fold_2_arg)]#.reset_index()
fold_3_arg_df = csv_df_arg_shuffled[csv_df_arg_shuffled['ID'].isin(fold_3_arg)]#.reset_index()
fold_4_arg_df = csv_df_arg_shuffled[csv_df_arg_shuffled['ID'].isin(fold_4_arg)]#.reset_index()
fold_5_arg_df = csv_df_arg_shuffled[csv_df_arg_shuffled['ID'].isin(fold_5_arg)]#.reset_index()

fold_1_polesie_df = csv_df_polesie_shuffled[csv_df_polesie_shuffled['ID'].isin(fold_1_polesie)]#.reset_index()
fold_2_polesie_df = csv_df_polesie_shuffled[csv_df_polesie_shuffled['ID'].isin(fold_2_polesie)]#.reset_index()
fold_3_polesie_df = csv_df_polesie_shuffled[csv_df_polesie_shuffled['ID'].isin(fold_3_polesie)]#.reset_index()
fold_4_polesie_df = csv_df_polesie_shuffled[csv_df_polesie_shuffled['ID'].isin(fold_4_polesie)]#.reset_index()
fold_5_polesie_df = csv_df_polesie_shuffled[csv_df_polesie_shuffled['ID'].isin(fold_5_polesie)]#.reset_index()

fold_1_isic_df = csv_df_isic_shuffled[csv_df_isic_shuffled['isic_id'].isin(fold_1_isic)]#.reset_index()
fold_2_isic_df = csv_df_isic_shuffled[csv_df_isic_shuffled['isic_id'].isin(fold_2_isic)]#.reset_index()
fold_3_isic_df = csv_df_isic_shuffled[csv_df_isic_shuffled['isic_id'].isin(fold_3_isic)]#.reset_index()
fold_4_isic_df = csv_df_isic_shuffled[csv_df_isic_shuffled['isic_id'].isin(fold_4_isic)]#.reset_index()
fold_5_isic_df = csv_df_isic_shuffled[csv_df_isic_shuffled['isic_id'].isin(fold_5_isic)]#.reset_index()



fold_1_combined_df = pd.concat([fold_1_vdr_df, fold_1_arg_df, fold_1_polesie_df, fold_1_isic_df])
fold_2_combined_df = pd.concat([fold_2_vdr_df, fold_2_arg_df, fold_2_polesie_df, fold_2_isic_df])
fold_3_combined_df = pd.concat([fold_3_vdr_df, fold_3_arg_df, fold_3_polesie_df, fold_3_isic_df])
fold_4_combined_df = pd.concat([fold_4_vdr_df, fold_4_arg_df, fold_4_polesie_df, fold_4_isic_df])
fold_5_combined_df = pd.concat([fold_5_vdr_df, fold_5_arg_df, fold_5_polesie_df, fold_5_isic_df])
fold_1_combined_df = fold_1_combined_df.sample(frac=1).reset_index(drop=True)
fold_2_combined_df = fold_2_combined_df.sample(frac=1).reset_index(drop=True)
fold_3_combined_df = fold_3_combined_df.sample(frac=1).reset_index(drop=True)
fold_4_combined_df = fold_4_combined_df.sample(frac=1).reset_index(drop=True)
fold_5_combined_df = fold_5_combined_df.sample(frac=1).reset_index(drop=True)


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
		self.N_CLASSES = N_CLASSES

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



from torchvision import transforms
#DATA AUGMENTATION
prob = 0.5
pipeline_transform = A.Compose([
	A.VerticalFlip(p=prob),
	A.HorizontalFlip(p=prob),
	A.RandomRotate90(p=prob),
	# A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),p=prob)
	])

#DATA NORMALIZATION
preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	transforms.Resize((224,224)),
])


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

	def __init__(self, dataset, indices=None, num_samples=None):
				
		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset)))             if indices is None else indices
			
		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices)             if num_samples is None else num_samples
			
		# distribution of classes in the dataset 
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1
				
		# weight for each sample
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
				   for idx in self.indices]
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):
		return dataset[idx,1]
				
	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

class Dataset_train(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.labels = labels
		self.list_IDs = list_IDs
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):
		
		# Select sample
		ID = self.list_IDs[index]
		# Load data and get label
		X = Image.open(ID)
		X = X.convert('RGB')

		X = np.asarray(X)
		y = self.labels[index]
		#data augmentation
		new_image = pipeline_transform(image=X)['image']
		new_image = np.asarray(new_image)
		#data transformation
		input_tensor = preprocess(X)
				
		return input_tensor, np.asarray(y)

class Dataset_test(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.list_IDs = list_IDs
		self.labels = labels
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):

		# Select sample
		ID = self.list_IDs[index]
		# Load data and get label
		X = Image.open(ID)
		X = X.convert('RGB')

		X = np.asarray(X)
		y = self.labels[index]

		#data transformation
		input_tensor = preprocess(X)
				
		return input_tensor, np.asarray(y)





for i in range(5):
	print("Running Fold", i+1, "/", 5)
	OUTPUT_PATH_FOLD = models_path + 'Fold_'+str(i) + os.sep
	create_dir(OUTPUT_PATH_FOLD)

	finetuned_strongly_supervised_model_path = OUTPUT_PATH_FOLD+'fully_supervised_model_strongly.pt'

	if i == 0:
		train_df = pd.concat([fold_1_combined_df, fold_2_combined_df, fold_3_combined_df, fold_4_combined_df], axis=0, ignore_index=True)
		test_df = fold_5_combined_df
	elif i == 1:
		train_df = pd.concat([fold_1_combined_df, fold_2_combined_df, fold_3_combined_df, fold_5_combined_df], axis=0, ignore_index=True)
		test_df = fold_4_combined_df
	elif i == 2:
		train_df = pd.concat([fold_1_combined_df, fold_2_combined_df, fold_4_combined_df, fold_5_combined_df], axis=0, ignore_index=True)
		test_df = fold_3_combined_df
	elif i == 3:
		train_df = pd.concat([fold_1_combined_df, fold_3_combined_df, fold_4_combined_df, fold_5_combined_df], axis=0, ignore_index=True)
		test_df = fold_2_combined_df
	elif i == 4:
		train_df = pd.concat([fold_2_combined_df, fold_3_combined_df, fold_4_combined_df, fold_5_combined_df], axis=0, ignore_index=True)
		test_df = fold_1_combined_df

	print(train_df['CLASS'].value_counts())
	print(test_df['CLASS'].value_counts())

	train_df.to_csv(OUTPUT_PATH_FOLD+'train_df.csv', index = False, header=True, columns=['IMAGE_PATH','CLASS'])
	test_df.to_csv(OUTPUT_PATH_FOLD+'validation_df.csv', index = False, header=True, columns=['IMAGE_PATH','CLASS'])

	train_dataset = train_df.values
	valid_dataset = test_df.values
	print(train_dataset)

	for i in range(len(train_dataset)):
		train_dataset[i,15] = int(train_dataset[i,15])
		train_dataset[i,1] = str(train_dataset[i,1])

	for i in range(len(valid_dataset)):
		valid_dataset[i,15] = int(valid_dataset[i,15])
		valid_dataset[i,1] = str(valid_dataset[i,1])
	


	# Parameters

	params_train = {'batch_size': int(BATCH_SIZE),
			#'shuffle': True,
			'sampler': ImbalancedDatasetSampler(train_dataset),
			'num_workers': 0}

	params_valid = {'batch_size': int(BATCH_SIZE),
			'shuffle': True,
			#'sampler': ImbalancedDatasetSampler(train_dataset),
			'num_workers': 0}


	max_epochs = int(EPOCHS)

	training_set = Dataset_train(train_dataset[:,1], train_dataset[:,15])
	training_generator = data.DataLoader(training_set, **params_train)

	validation_set = Dataset_test(valid_dataset[:,1], valid_dataset[:,15])
	validation_generator = data.DataLoader(validation_set, **params_valid)

	model = StudentModel()


	if PARALLEL == True:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	else:
		device = torch.device('cuda',DEV_GPU)


	# Find total parameters and trainable parameters
	total_params = sum(p.numel() for p in model.parameters())
	print(f'{total_params:,} total parameters.')
	total_trainable_params = sum(
		p.numel() for p in model.parameters() if p.requires_grad)
	print(f'{total_trainable_params:,} training parameters.')

	import torch.optim as optim
	criterion = torch.nn.CrossEntropyLoss()
	num_epochs = max_epochs
	epoch = 0
	early_stop_cont = 0
	EARLY_STOP_NUM = 10



	optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=True)
	model.to(device)


	best_loss_patch_level = 1000000000.0

	losses_train = []
	losses_valid_patches = []

	tot_batches_training = int(len(train_dataset) / int(BATCH_SIZE))


	print('Total batches training:', tot_batches_training)
	print(len(train_dataset) )


	while (epoch<num_epochs and early_stop_cont<EARLY_STOP_NUM):
		train_loss = 0.0

		y_true = []
		y_pred = []
		
		#accuracy for the outputs
		acc = 0.0
		
		is_best = False
		
		i = 0
		
		model.train()
		
		for inputs,labels in training_generator:
			inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)

			optimizer.zero_grad()

			logits, outputs = model(inputs, None)

			loss = criterion(logits, labels)

			loss.backward()
			optimizer.step()
			
			train_loss = train_loss + ((1 / (i+1)) * (loss.item() - train_loss))   

			outputs_np = outputs.cpu().data.numpy()
			labels_np = labels.cpu().data.numpy()
			outputs_np = np.argmax(outputs_np, axis=1)
			
			y_true = np.append(y_true, labels_np)
			y_pred = np.append(y_pred, outputs_np)
					
			i = i+1

			acc = metrics.accuracy_score(y_true, y_pred)
			kappa = metrics.cohen_kappa_score(y_true, y_pred)

			if (i%10==0):
				print("["+str(i)+"/"+str(tot_batches_training)+"], loss: " + str(train_loss))
			
		model.eval()
		
		print('[epoch %d] loss: %.4f, acc_train: %.4f, kappa_train: %.4f' %
				(epoch, train_loss, acc, kappa))

		losses_train.append(train_loss)

		print("evaluate")
		y_pred_val = []
		y_true_val = []
		
		valid_loss = 0.0

		with torch.no_grad():
			j = 0
			for inputs,labels in validation_generator:
				inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)

				logits, outputs = model(inputs, None)

				loss = criterion(logits, labels)
						
				valid_loss = valid_loss + ((1 / (j+1)) * (loss.item() - valid_loss))
				
				outputs_np = outputs.cpu().data.numpy()
				labels_np = labels.cpu().data.numpy()
				outputs_np = np.argmax(outputs_np, axis=1)
				
				y_pred_val = np.append(y_pred_val,outputs_np)
				y_true_val = np.append(y_true_val,labels_np)

				j = j+1

				
			acc_valid = metrics.accuracy_score(y_pred_val, y_true_val)
			kappa_valid = metrics.cohen_kappa_score(y_pred_val, y_true_val)

		print('[%d] valid loss: %.4f, acc_valid: %.4f, kappa_valid: %.4f' %
			(epoch + 1, valid_loss, acc_valid, kappa_valid))

		losses_valid_patches.append(valid_loss)

		if (best_loss_patch_level>valid_loss):
			print("previous best loss: " + str(best_loss_patch_level) + ", new best loss function: " + str(valid_loss))
			best_loss_patch_level = valid_loss
			is_best = True
			torch.save(model, finetuned_strongly_supervised_model_path)
			early_stop_cont = 0
		else:
			early_stop_cont = early_stop_cont+1
		
		epoch = epoch + 1
		
	print('Finished Training')