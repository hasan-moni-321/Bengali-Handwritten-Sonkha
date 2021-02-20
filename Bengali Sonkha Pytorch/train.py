import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2, pickle, glob, os, pickle

from sklearn.model_selection import train_test_split

import torch 
import torch.nn as nn 
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from torchvision import transforms

import model 
import models 
import dataset 
import engine
from Early_Stopping_Implementation import EarlyStopping

# Path of the datasets
train_img = "/home/hasan/Data Set/Bengali Digit/training-a"
valid_img = "/home/hasan/Data Set/Bengali Digit/training-b"
test_img = "/home/hasan/Data Set/Bengali Digit/training-d" 
train_labels = pd.read_csv("/home/hasan/Data Set/Bengali Digit/training-a.csv")
valid_labels = pd.read_csv("/home/hasan/Data Set/Bengali Digit/training-b.csv")
test_labels = pd.read_csv("/home/hasan/Data Set/Bengali Digit/training-d.csv")

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu' 



# Train data loading
#train_transf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
train_transf = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                    ])
train_data = dataset.ImageData(df = train_labels, data_dir = train_img, resize=50, transform = train_transf)
train_loader = DataLoader(dataset = train_data, batch_size = 64)


# Valid data loading
#valid_transf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
valid_data = dataset.ImageData(df = valid_labels, data_dir = valid_img, resize=50, transform = train_transf)
valid_loader = DataLoader(dataset = valid_data, batch_size = 64, shuffle=True, num_workers=2)


# Test data loading
#test_transf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                    ])
test_data = dataset.ImageData(df = test_labels, data_dir = test_img, resize=50, transform = transform_test)
test_loader = DataLoader(dataset = test_data, batch_size = 64, shuffle=False, num_workers=2) 



# loading model and device
#model = model.VGG("VGG16")
model = models.model4
model.to(device)

# loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
early_stopping = EarlyStopping(patience=10, verbose=True)


# Training The Model
epochs = 30
train_losses = []
valid_losses = []
for epoch in range(epochs):
    train_loss = engine.train(
                train_loader, 
                model, 
                optimizer, 
                device,
                criterion,
                epoch
                )
    train_losses.append(train_loss)

    valid_loss = engine.valid(
                valid_loader, 
                model, 
                criterion, 
                device,
                epoch,
                early_stopping
                )
    valid_losses.append(valid_loss) 


# Training and Validation loss graph
print("Final train losses is :", np.mean(train_losses))
print("Final validation losses is :", np.mean(valid_losses))



# Testing with test data
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        #test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Test Accuracy of the model on test images: {} %'.format(100 * correct / total))

