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

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=32)
parser.add_argument('-m', '--MODEL', help='model to use',type=str, default='densenet121')
parser.add_argument('-f', '--FOLD', help='fold to use',type=str, default='majority')
parser.add_argument('-t', '--TASK', help='task to perform',type=str, default='Multiclass')

args = parser.parse_args()

N_EXP = args.N_EXP
BATCH_SIZE = args.BATCH_SIZE
MODEL = args.MODEL
FOLD = args.FOLD
TASK = args.TASK

# print("TESTING FULLY SUPERVISED MODEL, N_EXP " + N_EXP)

def create_dir(directory):
	if not os.path.isdir(directory):
		try:
			os.mkdir(directory)
		except OSError:
			print ("Creation of the directory %s failed" % directory)
		else:
			print ("Successfully created the directory %s " % directory)


OUTPUT_PATH_ANNOTATIONS = '/home/jpdominguez/projects/BreslowTotal/src/semi_supervision/'
create_dir(OUTPUT_PATH_ANNOTATIONS)
OUTPUT_PATH_ANNOTATIONS = OUTPUT_PATH_ANNOTATIONS + 'annotations/'
create_dir(OUTPUT_PATH_ANNOTATIONS)
CSV_PATH = '/home/jpdominguez/projects/BreslowTotal/data/ISIC con filtro - to annotate/metadata.csv'
IMAGES_PATH = '/home/jpdominguez/projects/BreslowTotal/data/ISIC con filtro - to annotate/'


create_dir(OUTPUT_PATH_ANNOTATIONS)

csv_df = pd.read_csv(CSV_PATH)
csv_df['IMAGE_PATH'] = [IMAGES_PATH + x + '.JPG' for x in csv_df['isic_id']]
# print(csv_df['IMAGE_PATH'])
print('Images found: ', np.all([os.path.isfile(i) for i in csv_df['IMAGE_PATH']]))


test_dataset = csv_df.values

models_path = '/home/jpdominguez/projects/BreslowTotal/src/full_supervision/train/models/'
models_path = models_path+MODEL+'/' + TASK + '/' + 'N_EXP_'+str(N_EXP)+'/'

N_CLASSES=2
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
preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	transforms.Resize((224,224)),
])

class Dataset_annotate(data.Dataset):

	def __init__(self, list_IDs):

		self.list_IDs = list_IDs
		
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

		#data transformation
		input_tensor = preprocess(X)
				
		return input_tensor, ID, index


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
params_test = {'batch_size': BATCH_SIZE,
		  'shuffle': False,
		  #'sampler': ImbalancedDatasetSampler(test_dataset),
		  'num_workers': 0}




def annotate_csv(fold):

	print("Running Fold", fold+1, "/", 5)
	OUTPUT_PATH_FOLD = models_path + 'Fold_'+str(fold) + os.sep
	model_path = OUTPUT_PATH_FOLD+'fully_supervised_model_strongly.pt'


	model = torch.load(model_path)
	model.eval()

	testing_set = Dataset_annotate(test_dataset[:,25])
	testing_generator = data.DataLoader(testing_set, **params_test)

	y_pred = []
	output_preds_0 = []
	output_preds_1 = []
	output_preds_2 = []
	filenames = []

	tot_batches_to_annotate = int(len(testing_set) / int(BATCH_SIZE))
	print('Total batches:', tot_batches_to_annotate)

	with torch.no_grad():
		j = 0
		for inputs, filename, idx in testing_generator:
			inputs = inputs.to(device)

			logits, outputs = model(inputs, None)

			outputs_np = outputs.cpu().data.numpy()
			outputs_np = np.argmax(outputs_np, axis=1)
			y_pred = np.append(y_pred,outputs_np)

			outputs_np = outputs.cpu().data.numpy()
			output_preds_0 = np.append(output_preds_0, outputs_np[:, [0]])
			output_preds_1 = np.append(output_preds_1, outputs_np[:, [1]])
			if TASK == 'Multiclass':
				output_preds_2 = np.append(output_preds_2, outputs_np[:, [2]])
			outputs_np = np.argmax(outputs_np, axis=1)
			filenames = np.append(filenames,filename)
			
			j += 1
			if j % 10 == 0:
				print('Batch:', j, '/', tot_batches_to_annotate)

	preds_filename = OUTPUT_PATH_ANNOTATIONS + 'preds_raw_' + 'N_EXP_' + str(N_EXP) + '_Fold_' + str(fold) + '.csv'

	print(filenames.shape)
	print(output_preds_0.shape)
	print(output_preds_1.shape)
	if TASK == 'Multiclass':
		print(output_preds_2.shape)

	if TASK == 'Multiclass':
		File = {'filename':filenames,'y_pred_0':output_preds_0,'y_pred_1':output_preds_1,'y_pred_2':output_preds_2}
		df = pd.DataFrame(File,columns=['filename','y_pred_0','y_pred_1','y_pred_2'])
	elif TASK == 'InSitu' or TASK == 'Breslow':
		File = {'filename':filenames,'y_pred_0':output_preds_0,'y_pred_1':output_preds_1}
		df = pd.DataFrame(File,columns=['filename','y_pred_0','y_pred_1'])
	return df, preds_filename
	#np.savetxt(preds_filename, df.values, fmt='%.14f',delimiter=',')


if FOLD != 'majority':
	df, preds_filename = annotate_csv(int(FOLD))
	np.savetxt(preds_filename, df.values, fmt='%s',delimiter=',')

else:
	df_folds = []
	for i in range(5):
		df, _ = annotate_csv(i)
		df_folds.append(df)
	
	# get the majority vote for each image across the 5 folds
	df_majority = df_folds[0]
	for i in range(1,5):
		df_majority['y_pred_0'] = df_majority['y_pred_0'] + df_folds[i]['y_pred_0']
		df_majority['y_pred_1'] = df_majority['y_pred_1'] + df_folds[i]['y_pred_1']
		if TASK == 'Multiclass':
			df_majority['y_pred_2'] = df_majority['y_pred_2'] + df_folds[i]['y_pred_2']

	df_majority['y_pred_0'] = df_majority['y_pred_0'] / 5
	df_majority['y_pred_1'] = df_majority['y_pred_1'] / 5
	if TASK == 'Multiclass':
		df_majority['y_pred_2'] = df_majority['y_pred_2'] / 5

	if TASK == 'Breslow' or TASK == 'InSitu':
		# df_majority['y_pred_0'] = df_majority['y_pred_0'].apply(lambda x: 1 if x > 0.5 else 0)
		# df_majority['y_pred_1'] = df_majority['y_pred_1'].apply(lambda x: 1 if x > 0.5 else 0)
		df_majority['winner'] = df_majority[['y_pred_0', 'y_pred_1']].values.argmax(axis=1)
	elif TASK == 'Multiclass':
		df_majority['winner'] = df_majority[['y_pred_0', 'y_pred_1', 'y_pred_2']].values.argmax(axis=1)

	preds_filename = OUTPUT_PATH_ANNOTATIONS + 'preds_' + TASK + '_' + MODEL + '_N_EXP_' + str(N_EXP) + '_majority.csv'
	np.savetxt(preds_filename, df_majority.values, fmt='%s',delimiter=',')





