import argparse
import os
import cv2
import glob
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')
    parser.add_argument('--model', type=str, default='densenet121',
                        choices=[
                            'densenet121', 'resnet50', 'vgg16'
                        ],
                        help='model to use')
    parser.add_argument('--task', type=str, default='Breslow',
                        choices=[
                            'Breslow', 'InSitu', 'Multiclass'
                        ],
                        help='task to perform')
    parser.add_argument('--N_EXP', type=int, default=30,
                        choices=[
                            10, 20, 30
                        ],
                        help='N_EXP to use')
    parser.add_argument('--fold', type=int, default=1,
                        choices=[
                            0, 1, 2, 3, 4
                        ],
                        help='Fold to use')
    parser.add_argument('--training_method', type=str, default='full_supervision',
                        choices=[
                            'full_supervision', 'semi_supervision'
                        ],
                        help='Fold to use')
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }

    # SETTINGS

    imageNet_weights = True

    if args.task == 'Breslow' or args.task == 'InSitu':
        N_CLASSES = 2
    elif args.task == 'Multiclass':
        N_CLASSES = 3

    if args.training_method == 'fully_supervision':
        model_path = '/home/jpdominguez/projects/BreslowTotal/src/full_supervision/train/models/' + args.model + '/' + args.task + '/N_EXP_' + str(args.N_EXP) + '/Fold_' + str(args.fold) + '/fully_supervised_model_strongly.pt'
    elif args.training_method == 'semi_supervision':
        model_path = '/home/jpdominguez/projects/BreslowTotal/src/semi_supervision/train/models/' + args.model + '/' + args.task + '/N_EXP_' + str(args.N_EXP) + '/Fold_' + str(args.fold) + '/student_semisupervised.pt'
    # model_path = '/home/jpdominguez/projects/BreslowTotal/src/pytorch-grad-cam-master/fully_supervised_model_strongly.pt'
    # if args.model == 'densenet121':
    #     model_path = '/home/jpdominguez/projects/BreslowTotal/src/full_supervision/train/models/densenet121/Breslow/N_EXP_30/Fold_2/fully_supervised_model_strongly.pt'
    # elif args.model == 'resnet50':
    #     model_path = '/home/jpdominguez/projects/BreslowTotal/src/full_supervision/train/models/resnet50/Breslow/N_EXP_30/Fold_2/fully_supervised_model_strongly.pt'
    # elif args.model == 'vgg16':
    #     model_path = '/home/jpdominguez/projects/BreslowTotal/src/full_supervision/train/models/vgg16/Breslow/N_EXP_30/Fold_2/fully_supervised_model_strongly.pt'
    dataset_path = '/home/jpdominguez/projects/BreslowTotal/data/VdR_new_images_metadata_TOTEST/'
    
    output_dir = '/home/jpdominguez/projects/BreslowTotal/src/pytorch-grad-cam-master/output/'
    os.makedirs(output_dir, exist_ok=True)
    output_dir = output_dir + args.training_method + '/' 
    os.makedirs(output_dir, exist_ok=True)
    output_dir = output_dir + args.model + '/' 
    os.makedirs(output_dir, exist_ok=True)
    output_dir = output_dir + args.task + '/' 
    os.makedirs(output_dir, exist_ok=True)
    output_dir = output_dir + 'N_EXP_' + str(args.N_EXP) + '/'
    os.makedirs(output_dir, exist_ok=True)
    output_dir = output_dir + args.method + '/'
    os.makedirs(output_dir, exist_ok=True)
    output_dir = output_dir + 'Fold_' + str(args.fold) + '/'
    os.makedirs(output_dir, exist_ok=True)



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

        def forward(self, x):
                """
                In the forward function we accept a Tensor of input data and we must return
                a Tensor of output data. We can use Modules defined in the constructor as
                well as arbitrary operators on Tensors.
                """
                A = None
                m_binary = torch.nn.Sigmoid()
                m_multiclass = torch.nn.Softmax()

                dropout = torch.nn.Dropout(p=0.2)
                #spyder
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
                
                return  output_fcn 

    model = torch.load(model_path).to(torch.device(args.device)).eval()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    
    

    # layers = list(model.children())[-4]
    # print(list(model.children()))
    # print(list(model.children())[0])
    # print(model.children()[-4])

    # print(dict(model.named_modules()))
    # print(model.conv_layers[0].denseblock4.denselayer16.conv2)
    # print(model.layer4)


    # if dbg:
    #     print(model.layer4)
    # else:
    #     print(list(model.children()))

    # target_layers = [model.layer4]

    # densenet121 3 classes
    if args.model == 'densenet121':
        target_layers = [model.conv_layers[0].denseblock4.denselayer16.conv2]
    elif args.model == 'resnet50':
        target_layers = [model.conv_layers[7][2].conv3]
    elif args.model == 'vgg16':
        target_layers = [model.conv_layers[0][28]]
    # resnet50 2 classes
    # target_layers = [model.conv_layers[7]]


    # layers = list(net.children())[-4]

    images_path = glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True)
    for image_path in images_path:
        # print image name without the whole path and extension
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(image_name)

        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        rgb_img = cv2.resize(rgb_img, (224, 224))
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]).to(args.device)


        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [ClassifierOutputTarget(281)]
        # targets = [ClassifierOutputTarget(281)]
        targets = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                        target_layers=target_layers) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
        gb = gb_model(input_tensor, target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        
        cam_output_path = output_dir + image_name + '_' + args.method + '_cam.jpg'
        gb_output_path = output_dir + image_name + '_' + args.method + '_gb.jpg'
        cam_gb_output_path = output_dir + image_name + '_' + args.method + '_cam_gb.jpg'

        # cam_output_path = os.path.join(output_dir, image_name, f'{args.method}_cam.jpg')
        # gb_output_path = os.path.join(output_dir, image_name, f'{args.method}_gb.jpg')
        # cam_gb_output_path = os.path.join(output_dir, image_name, f'{args.method}_cam_gb.jpg')

        # print(f'CAM image saved at: {cam_output_path}')
        cv2.imwrite(cam_output_path, cam_image)
        cv2.imwrite(gb_output_path, gb)
        cv2.imwrite(cam_gb_output_path, cam_gb)


