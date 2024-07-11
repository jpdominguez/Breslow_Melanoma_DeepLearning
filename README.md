# Melanoma Breslow thickness classification using ensemble-based knowledge distillation with semi-supervised deep convolutional neural networks

<h2>Abstract</h2>

<p align="justify">
Background: melanoma is considered a global public health challenge and is responsible for more than 90\% of deaths related to skin cancer. Although the diagnosis of early melanoma is the main goal of dermoscopy, the discrimination between dermoscopic images of in situ and invasive melanomas can be a difficult task even for experienced dermatologists. The recent advances in artificial intelligence in the field of medical image analysis show that its application to dermoscopy with the aim of supporting and providing a second opinion to the medical expert could be of great interest.

Method: in this work, four datasets from different sources were used to train and evaluate deep learning models on in situ versus invasive melanoma classification and on Breslow thickness prediction. Supervised learning and semi-supervised learning using a multi-teacher ensemble knowledge distillation approach were considered and evaluated using a stratified 5-fold cross-validation scheme.

Results: the best models achieved AUCs of 0.6186±0.0410 and of 0.7501±0.0674 on the former and latter classification tasks, respectively. The best results were obtained using semi-supervised learning, with the best model achieving 0.7751 and 0.8164 AUC, respectively.

Conclusions: the obtained results show that semi-supervised learning could improve the performance of trained models on different melanoma classification tasks when compared to supervised learning. Deep learning-based automatic diagnosis systems could help supporting medical experts in their decision, serving as a second opinion or as a triage tool for medical centers.
</p>

<h2>Data</h2>

Data are available on request from the corresponding author. Images from Virgen del Rocío University Hospital (VRUH) Dataset are available upon request at <a href=https://institucional.us.es/breslowdataset>https://institucional.us.es/breslowdataset</a>.

<h2>Code</h2>

### 1. Supervised learning

The folder (src/full_supervision) includes scripts to train and test CNNs in a supervised manner. In particular, the main scripts are:

- src/full_supervision/train/train_supervised.py
  * -n (--N_EXP): experiment number (used as an identifier: in case multiple versions of the same model are trained, different folders will be created identified with N_EXP).
  * -b (--BATCH_SIZE): batch size to use.
  * -e (--EPOCHS): number of epochs used to train the model.
  * -m (--MODEL): CNN model to use. 4 options are available: densenet121, resnet50, vgg16 and inceptionv3.
  * -l (--LEARNING_RATE): learning rate to use.
  * -c (--NUM_CLASSES): number of diferent labels.
  * -t (--TASK): task to perform. The options avaiable are: Breslow (Breslow < 0.8 mm vs Breslow >= 0.8  mm), InSitu (Miv vs Mis), or Multiclass (Mis vs Miv with BT < 0.8 mm vs Miv with BT >= 0.8 mm).

- src/full_supervision/test/test_supervised.py
  * -n (--N_EXP): experiment number (used as an identifier: in case multiple versions of the same model are trained, different folders will be created identified with N_EXP).
  * -b (--BATCH_SIZE): batch size to use.
  * -m (--MODEL): CNN model to use. 4 options are available: densenet121, resnet50, vgg16 and inceptionv3.
  * -c (--NUM_CLASSES): number of diferent labels.
  * -t (--TASK): task to perform. The options avaiable are: Breslow (Breslow < 0.8 mm vs Breslow >= 0.8  mm), InSitu (Miv vs Mis), or Multiclass (Mis vs Miv with BT < 0.8 mm vs Miv with BT >= 0.8 mm).
  * -d (--DATASET): dataset to evaluate. The options available are: all (combined results for the whole test set), rocio (Breslow VRUH ddataset), Polesie (Polesie et al.), Argenciano (Kawahara et al.), ISIC (ISIC challenge dataset).

- src/full_supervision/test/test_external_data.py
  * -n (--N_EXP): experiment number (used as an identifier: in case multiple versions of the same model are trained, different folders will be created identified with N_EXP).
  * -b (--BATCH_SIZE): batch size to use.
  * -m (--MODEL): CNN model to use. 4 options are available: densenet121, resnet50, vgg16 and inceptionv3.
  * -c (--NUM_CLASSES): number of diferent labels.
  * -t (--TASK): task to perform. The options avaiable are: Breslow (Breslow < 0.8 mm vs Breslow >= 0.8  mm), InSitu (Miv vs Mis), or Multiclass (Mis vs Miv with BT < 0.8 mm vs Miv with BT >= 0.8 mm).

<h2>Acknowledgements</h2>

This work was partially supported by the Andalusian Regional Project (with FEDER support) DAFNE (US-1381619). This work was also partially supported by the EUROMELANOMA 2023 project from the Piel Sana Foundation of the Academia Española de Dermatología y Venereología.
