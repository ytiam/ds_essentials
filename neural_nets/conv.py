import pandas as pd
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from numpy import moveaxis
from tqdm import tqdm
from typing import Union

from torch.utils.data.dataloader import DataLoader
import torch
torch.cuda.empty_cache()
import gc
gc.collect()


# Generic Class for pytorch align framework to train a neural net with few basic operational modules like fit, predict, calculate_accuracy etc.

class NeuralNetBasic:
    
    def __init__(self, arch = None, X = None, y = None, lr = 0.001, weight_decay=0.00001, 
                 random_state: int = 32, test_size: float = 0.1, batch_size: int = 2,
                 optimizer = "adam", criterion = None):
        '''
        Initialize parameters

        Args:
        arch - The pytorch neural net architecture (e.g. GenreClassification define below)
        X - tensor array of feature set
        y - tensor array of labels
        lr - learning rate
        weight_decay - weight decay of the model
        random_state - random_state for the train-test split
        test_size - test set split percentage
        batch_size - training-validation batch size
        '''
        if criterion is None:
            raise ValueError("Please set a valid torch.nn.LOSS for criterion")
        
        self.arch = arch
        self.X = X
        self.y = y
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.test_size = test_size
        self.batch_size = batch_size

        torch.cuda.empty_cache()
        gc.collect()
        
        self.train_val_split()
        self.initiate_data_loaders()
        self.initialize_model(optimizer = optimizer, criterion = criterion)

    class conv_net(nn.Module):
        def __init__(self, in_channel: int, out_shape: int):
            super().__init__()
            self.network = nn.Sequential(nn.Conv2d(in_channel,16,kernel_size=5),
                          nn.BatchNorm2d(16),
                          nn.ReLU(),
                          nn.MaxPool2d(2,2),

                          nn.Conv2d(16,32,kernel_size=5),
                          nn.BatchNorm2d(32),
                          nn.ReLU(),
                          nn.MaxPool2d(2,2),

                          nn.Conv2d(32,64,kernel_size=5),
                          nn.BatchNorm2d(64),
                          nn.ReLU(),
                          nn.MaxPool2d(2,2),

                          nn.Conv2d(64,64,kernel_size=5),
                          nn.BatchNorm2d(64),
                          nn.ReLU(),
                          nn.MaxPool2d(2,2),

                          nn.Conv2d(64,64,kernel_size=5),
                          nn.BatchNorm2d(64),
                          nn.ReLU(),
                          nn.MaxPool2d(2,2),

                          nn.Flatten(),
                          nn.Linear(4096,256),
                          nn.ReLU(),
                          nn.Dropout(0.25),
                          nn.Linear(256,128),
                          nn.ReLU(),
                          nn.Linear(128,out_shape),
                          nn.Sigmoid()
                         )

        def forward(self, xb):
            return self.network(xb)
        
        
    def train_val_split(self):
        '''
        Make train validation data split from the given X and y
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=self.random_state, test_size=self.test_size)
        
        
    def initiate_data_loaders(self):
        '''
        Initialize the pytorch data loader objects for train and validation set
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_val_split()
        
        train_dat    = torch.utils.data.TensorDataset(torch.tensor(self.X_train).to(self.device), torch.tensor(self.y_train).to(self.device))
        self.train_loader = torch.utils.data.DataLoader(train_dat, batch_size = self.batch_size, shuffle = True)

        val_dat    = torch.utils.data.TensorDataset(torch.tensor(self.X_test).to(self.device), torch.tensor(self.y_test).to(self.device))
        self.val_loader = torch.utils.data.DataLoader(val_dat, batch_size = self.batch_size, shuffle = False)
    
    
    def initialize_model(self, optimizer: str = "adam", criterion = None):
        '''
        Initialize the model with optimizer and loss criterion

        Args:
        optimizer - An optimizer of torch.optim class. Like, Adam, SGD etc
        criterion - A loss criteria of torch.nn class. lile, BCEloss, CrossEntropy etc
        '''
        if self.arch is None:
            self.model = self.conv_net(in_channel=self.X_train.shape[1], out_shape=self.y_train.shape[1])
        else:
            print("From Outside")
            self.model = self.arch(in_channel=self.X_train.shape[1], out_shape=self.y_train.shape[1])
            
        # optimizer and the loss function definition 
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        
        self.criterion = criterion

        #pin to gpu
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        
    def fit(self, epoch_ = 10):
        '''
        Function to Fit the data to the model

        Args:
        epoch_ - model training epochs
        optimizer - An optimizer of torch.optim class. Like, Adam, SGD etc
        criterion - A loss criteria of torch.nn class. lile, BCEloss, CrossEntropy etc 

        Out:
        Returns the trained model
        '''
        
        min_valid_loss = np.inf
        for epoch in range(epoch_):
            self.train_loss = 0
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch,0):
                    self.model.train()
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    #zero grad
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    inputs = inputs.type(torch.cuda.FloatTensor)
                    labels = labels.type(torch.cuda.FloatTensor)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    self.train_loss += loss.item()

                    tepoch.set_postfix(train_loss=loss.item(),Epoch=epoch)

                self.valid_loss = 0.0
                self.model.eval()     # Optional when not using Model Specific layer
                for val_inputs, val_labels in self.val_loader:
                    # Transfer Data to GPU if available
                    val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)
                    
                    # Convert the tensor type to Float
                    val_inputs = val_inputs.type(torch.cuda.FloatTensor)
                    val_labels = val_labels.type(torch.cuda.FloatTensor)
                    
                    # Forward Pass
                    target = self.model(val_inputs)
                    
                    # Find the Loss
                    loss = self.criterion(target,val_labels)
                    
                    # Calculate Loss
                    self.valid_loss += loss.item()

                print(f'Epoch {epoch+1} \t\t Training Loss: {self.train_loss / len(self.train_loader)} \t\t Validation Loss: {self.valid_loss / len(self.val_loader)}')

                if min_valid_loss > self.valid_loss:
                    print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{self.valid_loss:.6f}) \t Saving The Model')
                    min_valid_loss = self.valid_loss
                    torch.save(self.model.state_dict(), 'saved_model.pth')
                    
        return self.model
    
    
    def val_loss(self,data_loader = None):
        '''
        Calculate average loss on the given data

        Args:
        data_loader - A pytorch dataloader object. Default value is None. when None, the function will consider validation loader as default
        data_loader

        Out:
        Average loss
        '''
        
        if data_loader is None:
            data_loader = self.val_loader
        
        total = 0
        total_val_loss = 0
        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                images = images.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.FloatTensor)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total += labels.size(0)
                total_val_loss += loss.item()*labels.size(0)
        return total_val_loss/total
    
    
    def calculate_accuracy(self, data_loader = None):
        '''
        Calculate model accuracy on the given data

        Args:
        data_loader - A pytorch dataloader object. Default value is None. when None, the function will consider validation loader as default
        data_loader

        Out:
        Model accuracy on the given data
        '''
        if data_loader is None:
            flag = None
            data_loader = self.val_loader # default set to validation data
            
        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                images = images.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.FloatTensor)

                # number of samples and number of classes
                N, C = labels.shape 

                outputs = self.model(images)
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0
                total += labels.size(0)
                correct += (outputs == labels).sum().item()/C
        
        acc = round((100 * correct)/ total,2)
        
        if flag is None:
            print(f'Validation Accuracy is {acc:.2f}')
        return acc
    
    
    def predict(self, x = None):
        '''
        Calculate model prediction for the given data x

        Args:
        x - data in numpy array format

        Out:
        Model Prediction for the give data in numpy array format
        '''
        
        if x is None:
            raise ValueError("Pass a valid x on which prediction needs to be done")
        
        all_pred = []
        
        for vec in x:
            _dat    = torch.utils.data.TensorDataset(torch.tensor(np.array([vec])).to(self.device))
            _loader = torch.utils.data.DataLoader(_dat)

            with torch.no_grad():
                for im in _loader:

                    im = im[0].to(self.device)

                    im = im.type(torch.cuda.FloatTensor)

                    pred = self.model(im)
                    pred[pred >= 0.5] = 1
                    pred[pred < 0.5] = 0
                    #mask_pred = pred > 0
                    all_pred.append(pred[0].cpu())
        return all_pred
    

    ## Example Architecture. New Architectures need to be defined in the same fashion and the class must needs to include a forward function

    class GenreClassification(nn.Module):
        def __init__(self, in_channel: int, out_shape: int):
            super().__init__()
            self.network = nn.Sequential(nn.Conv2d(in_channel,16,kernel_size=5),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),

                        nn.Conv2d(16,32,kernel_size=5),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                                        
                        nn.Conv2d(32,64,kernel_size=5),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        
                        nn.Conv2d(64,64,kernel_size=5),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        
                        nn.Conv2d(64,64,kernel_size=5),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        
                        nn.Flatten(),
                        nn.Linear(4096,256),
                        nn.ReLU(),
                        nn.Dropout(0.25),
                        nn.Linear(256,128),
                        nn.ReLU(),
                        nn.Linear(128,out_shape),
                        nn.Sigmoid()
                        )
        
        def forward(self, xb):
            return self.network(xb)