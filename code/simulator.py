'''
simulator.py

Description: Code for running simulation to train the model, and saving results

Date Created: January 02, 2020

Revisions:
  - Jan 08, 2020:
      > typo: in self.types = [...] fixed (LFEEXPT -> LFE)
      > dataset filepaths have been replaced with config file inputs
  - Jan 05, 2020:
      > modify training code to save figures every 50 epochs
      > modify training code and config file to allow adjustable print and save plot frequencies
      > add option to delete results if exception encountered
      > modify training code to add option of showing plots
      > add saving of bar plots of final accuracy
  - Jan 04, 2020:
      > modify config file to better configure optimizer settings
      > modify config file to also store filepaths to dataset files
      > remove .float when extracting data from dataloader, this is now done when data is first loaded into dataset
      > make copy of config file in results folder
      > add weight decay
  - Jan 02, 2020: Multiple revisions, see below
      > create simulator class
      > migrate code to import data from plaut_model.ipynb
      > create configuration file to set up simulation

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import shutil

import configparser

from plaut_dataset import plaut_dataset
from model import plaut_net
from helpers import *

class simulator():
    def __init__(self):
        # set manual seed
        torch.manual_seed(1)
        
        # Load configuration file
        self.config = configparser.ConfigParser()
        self.config.read("config.cfg")

        # load dataset
        self.load_data()
        
        # Define word types to calculate accuracy for
        self.types = ["HEC", "HRI", "HFE", "LEC", "LFRI", "LFE"] # calculate accuracy of these types
        self.anc_types = ["ANC_REG", "ANC_EXC", "ANC_AMB"]
        self.probe_types = ["PRO_REG", "PRO_EXC", "PRO_AMB"]
            
        
    def load_data(self):
        '''
        IMPORTING DATASET
        > import dataset from csv files
        > multiple csv files can be concatenated in format: ["filename1", True/False, "filename2", True/False, ...]
        > True/False indicates whether to override frequencies to log(2)
        '''
        self.plaut_ds = plaut_dataset([self.config['dataset']['plaut'], False])
        self.anc_ds = plaut_dataset([self.config['dataset']['anchor'], False])
        self.probe_ds = plaut_dataset([self.config['dataset']['probe'], False])
        self.plaut_anc_ds = plaut_dataset([self.config['dataset']['plaut'], True, self.config['dataset']['anchor'], False])
        # Note: adding "True" after a filepath changes frequencies to ln(2), "False" leaves frequencies as is
        
        '''
        INITIALIZE DATALOADERS
        '''
        self.plaut_loader = DataLoader(self.plaut_ds, batch_size=len(self.plaut_ds), num_workers=0)
        self.anc_loader = DataLoader(self.anc_ds, batch_size=len(self.anc_ds), num_workers=0)
        self.probe_loader = DataLoader(self.probe_ds, batch_size=len(self.probe_ds), num_workers=0)
        self.plaut_anc_loader = DataLoader(self.plaut_anc_ds, batch_size=len(self.plaut_anc_ds), num_workers=0)
    
    
    def train(self):
        try: # run training function
            self.train_function()
        except KeyboardInterrupt: # if interrupted by keyboard
            print("Training Interrupted by User.")
            pass
        except: # raise any other error
            raise
        else: # no exception
            print("Training Completed!")
            save_notes(self.rootdir)
            return None
        if input("An exception occured. Delete plots and checkpoints? [y/n] \n  > ").lower() in ['y', 'yes']:
            if input("Are you sure? [y/n] \n  > ").lower() in ['y', 'yes']:
                shutil.rmtree(self.rootdir)
                print("Simulation results deleted.")
                return None
        save_notes(self.rootdir)
        print("Simulation results saved.")
        return None
    
    def train_function(self):
        # Initialize model
        self.model = plaut_net()
        
        # Create folder to store results
        self.rootdir = make_folder()
        print("Test Results will be stored in: ", self.rootdir)     
        
        # Make a copy of config file in results folder
        shutil.copyfile("config.cfg", self.rootdir+"/config.cfg")

        # Load optimizer settings from configuration file
        starts, optims, lrates, momenta, wds = [], [], [], [], []
        i = 1
        while True:
            try:
                starts.append(self.config['part'+str(i)]['start'])
                optims.append(self.config['part'+str(i)]['optim'])
                lrates.append(self.config['part'+str(i)]['lr'])
                momenta.append(self.config['part'+str(i)]['momentum'])
                wds.append(self.config['part'+str(i)]['wd'])
                i = i + 1
            except:
                break
        
        # load print and save frequency from config file
        print_freq = int(self.config['setup']['print_freq'])
        save_freq = int(self.config['setup']['save_freq'])
        
        # define loss function
        criterion = nn.BCELoss(reduction='none')

        # Initialize arrays to store epochs, train loss
        epochs, losses = [], []

        # Initialize arrays to store accuracy of plaut dataset, anchors, probes
        acc = [[] for i in self.types]
        anc_acc = [[] for i in self.anc_types]
        probe_acc = [[] for i in self.probe_types]
        
        data_loader = self.plaut_loader # set initial dataloader to not have anchors

        for epoch in range(int(self.config['setup']['total_epochs'])): # train for specified # of epochs
            # start timer
            epoch_time = time.time()
            
            # switch optimizer after specified # of epochs
            if len(starts) > 0 and epoch == int(starts[0]):
                if optims[0] == 'Adam':
                    optimizer = optim.Adam(self.model.parameters(), lr=float(lrates[0]), weight_decay=float(wds[0]))
                elif optims[0] == 'SGD':
                    optimizer = optim.SGD(self.model.parameters(), lr=float(lrates[0]), momentum=float(momenta[0]), weight_decay=float(wds[0]))
                else:
                    print("ERROR: Use either Adam or SGD optimizer")
                    return None
                
                optimizer.zero_grad() # zero the gradients
                
                for i in [starts, optims, lrates, momenta, wds]:
                    i.pop(0) # optimizer changed, pop first item off of arrays
            
            # after specified # of epochs, add anchors
            if epoch == int(self.config['setup']['anchor_epoch']): 
                data_loader = self.plaut_anc_loader
                
            avg_loss = 0 # initialize avg loss
            for i, data in enumerate(data_loader): 
                # extract frequency, inputs, labels
                freq = data["frequency"].view(-1, 1) # reshape to [batch_size x 1] to match output size
                inputs = data["graphemes"]
                labels = data["phonemes"]

                #forward pass + backward pass + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss = loss*freq
                loss = loss.mean() # scale loss by frequency, then find mean
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

            # print stats every X epochs
            if epoch % print_freq == print_freq - 1:
                print("[EPOCH %d] \t loss: %.6f \t time: %.4f \r" % (epoch+1, avg_loss, time.time()-epoch_time))

            # save loss and accuracy plots every X epochs
            if epoch % save_freq == save_freq - 1:
                make_plot(epochs, [losses], ["Train Loss"], "Epoch", "Loss", "Training Loss", save=True, filepath=self.rootdir+"/Training Loss/epoch_"+str(epoch+1)+".jpg", show=True)
                make_plot(epochs, acc, self.types, "Epoch", "Accuracy", "Training Accuracy", save=True, filepath=self.rootdir+"/Training Accuracy/epoch_"+str(epoch+1)+".jpg", show=True)
                make_plot(epochs, anc_acc, self.anc_types, "Epoch", "Accuracy", "Anchor Accuracy", save=True, filepath=self.rootdir+"/Anchor Accuracy/epoch_"+str(epoch+1)+".jpg", show=True)
                make_plot(epochs, probe_acc, self.probe_types, "Epoch", "Accuracy", "Probe Accuracy", save=True, filepath=self.rootdir+"/Probe Accuracy/epoch_"+str(epoch+1)+".jpg", show=True)

        # plot final loss and accuracy line plots and save
        make_plot(epochs, [losses], ["Train Loss"], "Epoch", "Loss", "Training Loss", save=True, filepath=self.rootdir+"/Training Loss Final.jpg", show=True)
        make_plot(epochs, acc, self.types, "Epoch", "Accuracy", "Training Accuracy", save=True, filepath=self.rootdir+"/Training Accuracy Final.jpg", show=True)
        make_plot(epochs, anc_acc, self.anc_types, "Epoch", "Accuracy", "Anchor Accuracy", save=True, filepath=self.rootdir+"/Anchor Accuracy Final.jpg", show=True)
        make_plot(epochs, probe_acc, self.probe_types, "Epoch", "Accuracy", "Probe Accuracy", save=True, filepath=self.rootdir+"/Probe Accuracy Final.jpg", show=True)
        
        # plot final accuracy bar plots and save
        make_bar(self.types, [i[-1] for i in acc], "Category", "Accuracy", "Final Training Accuracy", save=True, filepath=self.rootdir+"/Training Accuracy Bar.jpg", show=True)
        make_bar(self.anc_types, [i[-1] for i in anc_acc], "Category", "Accuracy", "Final Anchor Accuracy", save=True, filepath=self.rootdir+"/Anchor Accuracy Bar.jpg", show=True)
        make_bar(self.probe_types, [i[-1] for i in probe_acc], "Category", "Accuracy", "Final Probe Accuracy", save=True, filepath=self.rootdir+"/Probe Accuracy Bar.jpg", show=True)
        
