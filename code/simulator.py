'''
model.py

Description: Define the model architecture

Date Created: January 02, 2020

Revisions:
  - Jan 02, 2020: Multiple revisions, see below
      > create simulator class
      > migrate code to import data from plaut_model.ipynb
      > create configuration file to set up simulation

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import configparser

from plaut_dataset import plaut_dataset
from model import plaut_net
from helpers import *

class simulator():
    def __init__(self):
        self.load_data() # load dataset
        
        # Define word types to calculate accuracy for
        self.types = ["HEC", "HRI", "HFE", "LEC", "LFRI", "LFEEXPT"] # calculate accuracy of these types
        self.anc_types = ["ANC_REG", "ANC_EXC", "ANC_AMB"]
        self.probe_types = ["PRO_REG", "PRO_EXC", "PRO_AMB"]
            
        
    def load_data(self):
        '''
        IMPORTING DATASET
        > import dataset from csv files
        > multiple csv files can be concatenated in format: ["filename1", True/False, "filename2", True/False, ...]
        > True/False indicates whether to override frequencies to log(2)
        '''
        self.plaut_ds = plaut_dataset(["../dataset/plaut_dataset.csv", False])
        self.anc_ds = plaut_dataset(["../dataset/anchors.csv", False])
        self.probe_ds = plaut_dataset(["../dataset/probes.csv", False])
        self.plaut_anc_ds = plaut_dataset(['../dataset/plaut_dataset.csv', True, '../dataset/anchors.csv', False])
        
        '''
        INITIALIZE DATALOADERS
        '''
        self.plaut_loader = DataLoader(self.plaut_ds, batch_size=len(self.plaut_ds), num_workers=0)
        self.anc_loader = DataLoader(self.anc_ds, batch_size=len(self.anc_ds), num_workers=0)
        self.probe_loader = DataLoader(self.probe_ds, batch_size=len(self.probe_ds), num_workers=0)
        self.plaut_anc_loader = DataLoader(self.plaut_anc_ds, batch_size=len(self.plaut_anc_ds), num_workers=0)
    
    def train(self, lr=0.1):
        # Initialize model
        torch.manual_seed(1) # 2.
        self.model = plaut_net() # 3.
        
        # Create folder to store results
        self.rootdir = make_folder() # 4. 
        print("Test Results will be stored in: ", self.rootdir)     
        
        # Load configuration file
        config = configparser.ConfigParser()
        config.read("config.cfg")
        
        # define initial loss function and optimizer
        criterion = nn.BCELoss(reduction='none')
        if config['initial']['optim'] == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=float(config['initial']['lr']))
        elif config['initial']['optim'] == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=float(config['initial']['lr']), momentum=float(config['initial']['momentum']))
        else:
            print("ERROR: Use either Adam or SGD optimizer")
            return None

        # Initialize arrays to store epochs, train loss
        epochs, losses = [], []

        # Initialize arrays to store accuracy of plaut dataset, anchors, probes
        acc = [[] for i in self.types]
        anc_acc = [[] for i in self.anc_types]
        probe_acc = [[] for i in self.probe_types]
        
        data_loader = self.plaut_loader # set initial dataloader to not have anchors

        for epoch in range(int(config['setup']['total_epochs'])): # train for specified # of epochs
            
            # switch optimizer after specified # of epochs
            if epoch == int(config['setup']['inital_epochs']):
                if config['final']['optim'] == 'Adam':
                    optimizer = optim.Adam(self.model.parameters(), lr=float(config['final']['lr']))
                elif config['final']['optim'] == 'SGD':
                    optimizer = optim.SGD(self.model.parameters(), lr=float(config['final']['lr']), momentum=float(config['final']['momentum']))
                else:
                    print("ERROR: Use either Adam or SGD optimizer")
                    return None
            
            # after specified # of epochs, add anchors
            if epoch == int(config['setup']['anchor_epoch']): 
                data_loader = self.plaut_anc_loader
                

            avg_loss = 0 # initialize avg loss
            for i, data in enumerate(data_loader): 
                # extract frequency, inputs, labels
                freq = data["frequency"].float().view(-1, 1) # reshape to [batch_size x 1] to match output size
                inputs = data["graphemes"].float()
                labels = data["phonemes"].float()

                #forward pass + backward pass + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss = (loss*freq).mean() # scale loss by frequency, then find mean
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # calculate loss and save loss to array
            losses.append(avg_loss)
            epochs.append(epoch+1)

            # calculate accuracy over the different types for plaut dataset
            temp_acc = get_accuracy(self.model, self.plaut_loader, self.types, vowels_only=True)
            for i in range(len(self.types)):
                acc[i].append(temp_acc[i])

            # calculate accuracy over the different types for anchors
            temp_acc = get_accuracy(self.model, self.anc_loader, self.anc_types, vowels_only=True)
            for i in range(len(self.anc_types)):
                anc_acc[i].append(temp_acc[i])

            # calculate accuracy over the different types for probes
            temp_acc = get_accuracy(self.model, self.probe_loader, self.probe_types, vowels_only=True)
            for i in range(len(self.probe_types)):
                probe_acc[i].append(temp_acc[i])

            # print stats every 5 epochs
            if epoch % 1 == 0:
                print("[EPOCH %d] loss: %.6f" % (epoch+1, avg_loss))

            # plot loss every 50 epochs
            if epoch % 50 == 49:
                make_plot(epochs, [losses], ["Train Loss"], "Epoch", "Loss", "Training Loss")
                make_plot(epochs, acc, self.types, "Epoch", "Accuracy", "Training Accuracy")
                make_plot(epochs, anc_acc, self.anc_types, "Epoch", "Accuracy", "Anchor Accuracy")
                make_plot(epochs, probe_acc, self.probe_types, "Epoch", "Accuracy", "Probe Accuracy")


        # plot final loss curve and save
        plt.figure()
        plt.title("Training Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(losses, label="Training Loss")
        plt.savefig(self.rootdir+"/lossplot_final.png", dpi=150)
        plt.close()     
