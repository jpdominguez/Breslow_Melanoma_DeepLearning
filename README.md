# Melanoma Breslow thickness classification using ensemble-based knowledge distillation with semi-supervised deep convolutional neural networks


Background: melanoma is considered a global public health challenge and is responsible for more than 90\% of deaths related to skin cancer. Although the diagnosis of early melanoma is the main goal of dermoscopy, the discrimination between dermoscopic images of in situ and invasive melanomas can be a difficult task even for experienced dermatologists. The recent advances in artificial intelligence in the field of medical image analysis show that its application to dermoscopy with the aim of supporting and providing a second opinion to the medical expert could be of great interest.

Method: in this work, four datasets from different sources were used to train and evaluate deep learning models on in situ versus invasive melanoma classification and on Breslow thickness prediction. Supervised learning and semi-supervised learning using a multi-teacher ensemble knowledge distillation approach were considered and evaluated using a stratified 5-fold cross-validation scheme.

Results: the best models achieved AUCs of 0.6186±0.0410 and of 0.7501±0.0674 on the former and latter classification tasks, respectively. The best results were obtained using semi-supervised learning, with the best model achieving 0.7751 and 0.8164 AUC, respectively.

Conclusions: the obtained results show that semi-supervised learning could improve the performance of trained models on different melanoma classification tasks when compared to supervised learning. Deep learning-based automatic diagnosis systems could help supporting medical experts in their decision, serving as a second opinion or as a triage tool for medical centers.
