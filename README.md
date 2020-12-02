 ## Bengali-Handwritten-Sonkha
 
 ### About Dataset
 This is a classification project! Where I used Bengali handwritten Sonkha(means: Number of Bengali Language) dataset. Dataset have more than two hundred thousands 
 train images and more than seventeen thousands test images.Ten different classes of the images. Every image size is 32, 32 
 
 ### Bengali Sonkha Tensorflow
 In this project I used end-to-end opensource platform Tensorflow. For Data preprocessing I used cv2 for example GaussianBlur, AddWeight and alos added some kernel           to the image to get specific color of images to train properly.                                                                                                      
 I have built convolutional neural network model, because it is the image classification. I took 20% validation data from the train dataset. In this model I used Adam optimizer where learing rate is "1e-3"
 
 #### Accuracy and Loss
 | Train_Validation | Accuracy | Loss |
 | --- | --- | --- |
 | Train | .9955 | 0.0161 |
 | Validation | 0.9826 | 0.0885 |
 
 
  #### Future Update
  1. For image processing you can use albumentation, imgaug library those are very helpful for image during training. 
  1. Use of pretrained model like resenet34, resnet50, efficientnetb4 those gives very good result 
  
  
  ### Bengali Sonkha Pytorch
  Here I used opensource pytorch scientific library. I used custom-dataset and for image transformation used torchvision transforms.                            
  For model used VGG16 and this model is modified by building the layer.  CrossEntropyLoss and Adam optimizer are used 
  
  #### Accuracy and Loss
  | Train_Validation | Loss | Accuracy|
  | --- | --- | --- |
  | Train | 0.014 | |
  | Validation | 0.011 |
  | Test | .8901| 
  
  
  
