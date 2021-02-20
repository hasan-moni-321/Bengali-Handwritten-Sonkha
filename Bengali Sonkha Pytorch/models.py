import torch
import torch.nn as nn
import torchvision
from torchvision import models


# resnet18 model
def resnet18_model(num_class):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, num_class)
    return model_ft


# resnet34 model
def resnet34_model(num_class):
    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, num_class)
    return model_ft


# resnet50 model
def resnet50_model(num_class):
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, num_class)
    return model_ft


# inception_v3 model
def inception_model(num_class):
    model_ft = models.inception_v3(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, num_class)
    return model_ft


# googlenet model
def googlenet_model(num_class):
    model_ft = models.googlenet(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, num_class)
    return model_ft


# shufflenet_v2_x1_0 model
def shufflenet_model(num_class):
    model_ft = models.shufflenet_v2_x1_0(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, num_class)
    return model_ft


# resnext50_32x4d
def resnext50_32_model(num_class):
    model_ft = models.resnext50_32x4d(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, num_class)
    return model_ft


# wide_resnet50_2
def wide_resnet_model(num_class):
    model_ft = models.wide_resnet50_2(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, num_class)
    return model_ft


###################################################
# Calling the pytorch vision model
###################################################
model1 = resnet18_model(10)
model2 = resnet34_model(10)
model3 = resnet50_model(10)
model4 = inception_model(10)
model5 = googlenet_model(10)
model6 = shufflenet_model(10)
model7 = resnext50_32_model(10)
model8 = wide_resnet_model(10)
