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

print( torch.cuda.current_device())
print( torch.cuda.device_count())


import warnings
warnings.filterwarnings('ignore')


#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)
parser.add_argument('-e', '--EPOCHS', help='number of epochs',type=int, default=15)
parser.add_argument('-m', '--MODEL', help='model to use',type=str, default='densenet121')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
EPOCHS = args.EPOCHS
MODEL = args.MODEL


PARALLEL = False
DEV_GPU = 0

print("TRAINING FULLY SUPERVISED NETWORK, N_EXP " + N_EXP_str)

def create_dir(directory):
	if not os.path.isdir(directory):
		try:
			os.mkdir(directory)
		except OSError:
			print ("Creation of the directory %s failed" % directory)
		else:
			print ("Successfully created the directory %s " % directory) 


pretrained_models_path = 'E:\\Breslow\\src\\pytorch\\experiments\\Breslow\\full_supervision\\train\\models\\' +MODEL+'\\' + 'N_EXP_'+N_EXP_str+'\\'


models_path = 'E:\\Breslow\\src\\pytorch\\experiments\\Breslow\\semi_supervision\\train\\models\\'
create_dir(models_path)

models_path = models_path+MODEL+'\\'
create_dir(models_path)

models_path = models_path+'N_EXP_'+N_EXP_str+'\\'
create_dir(models_path) 


CSV_PATH_PSEUDO = 'E:\\Breslow\\src\\pytorch\\experiments\\Breslow\\full_supervision\\test\\teacher_annotations\\preds_raw_N_EXP_3_majority.csv'


# PSEUDO-LABELS

csv_df_pseudo = pd.read_csv(CSV_PATH_PSEUDO, header=None)
# remove column 0 from csv_df_pseudo_shuffled
csv_df_pseudo.columns = ['IMAGE_PATH', 'label', 'BRESLOW_processed']
csv_df_pseudo = csv_df_pseudo.drop(csv_df_pseudo.columns[1], axis=1)
# csv_df_pseudo = csv_df_pseudo.drop(csv_df_pseudo.columns[1], axis=1)


csv_df_pseudo_shuffled = csv_df_pseudo.sample(frac = 1, random_state= np.random.RandomState()).reset_index()
csv_df_pseudo_shuffled_iloc = csv_df_pseudo_shuffled.iloc[:][:]

print('Pseudo-labels:')
print('Images found:', len(csv_df_pseudo_shuffled_iloc), '--', np.all([os.path.isfile(i) for i in csv_df_pseudo_shuffled_iloc.values[:,1]]))
print('Breslow < 0.8:', len(csv_df_pseudo_shuffled[csv_df_pseudo_shuffled.BRESLOW_processed == 0]))
print('Breslow >= 0.8:', len(csv_df_pseudo_shuffled[csv_df_pseudo_shuffled.BRESLOW_processed == 1]))
print('------------------------')



N_CLASSES = 2
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
		self.N_CLASSES = 2

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
	A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),p=prob)
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
		
		# print(self.list_IDs[index])
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
		#print(ID)
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

	models_path_fold = models_path + 'Fold_'+str(i) + os.sep
	create_dir(models_path_fold)

	finetuned_strongly_supervised_model_path = models_path_fold+'fully_supervised_model_strongly.pt'

	pretrained_models_path_fold = pretrained_models_path + 'Fold_'+str(i) + os.sep

	train_strong_annotations_df = pd.read_csv(pretrained_models_path_fold + 'train_df.csv')
	test_strong_annotations_df = pd.read_csv(pretrained_models_path_fold + 'validation_df.csv')

	train_df = pd.concat([train_strong_annotations_df, csv_df_pseudo_shuffled], axis=0, ignore_index=True)
	train_df = train_df.sample(frac = 1, random_state= np.random.RandomState()).reset_index()
	test_df = test_strong_annotations_df

	print(train_strong_annotations_df)
	print(csv_df_pseudo_shuffled)


	print(train_df['BRESLOW_processed'].value_counts())
	print(test_df['BRESLOW_processed'].value_counts())


	train_df.to_csv(finetuned_strongly_supervised_model_path+'train_df.csv', index = False, header=True, columns=['IMAGE_PATH','BRESLOW_processed'])
	test_df.to_csv(finetuned_strongly_supervised_model_path+'validation_df.csv', index = False, header=True, columns=['IMAGE_PATH','BRESLOW_processed'])

	train_dataset = train_df.values
	valid_dataset = test_df.values

	# for i in range(len(train_dataset)):
	# 	train_dataset[i,16] = int(train_dataset[i,16])
	# 	train_dataset[i,2] = str(train_dataset[i,2])

	# for i in range(len(valid_dataset)):
	# 	valid_dataset[i,16] = int(valid_dataset[i,16])
	# 	valid_dataset[i,2] = str(valid_dataset[i,2])
	


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


	training_set = Dataset_train(train_dataset[:,1], train_dataset[:,2])
	training_generator = data.DataLoader(training_set, **params_train)

	validation_set = Dataset_test(valid_dataset[:,0], valid_dataset[:,1])
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
	EARLY_STOP_NUM = 5



	optimizer = optim.Adam(model.parameters(),lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=True)
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
