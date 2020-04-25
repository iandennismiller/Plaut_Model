'''
simulator.py

Description: Code for running simulation to train the model, and saving results

Date Created: January 02, 2020

Revisions:
  - Mar 30, 2020:
      > move save_checkpoint function to helpers.py
      > add testing function
      > move load_data to be executed in train function rather than init function
  - Mar 03, 2020:
      > update usage of plaut_dataset function
      > update file label to allow manual override of anchor dilation
  - Feb 23, 2020:
      > update loss calculation to remove scaling:
          BEFORE:                                            AFTER:
          loss = loss*freq                                   loss = loss*freq
          loss = loss.sum()/(freq.sum()*loss.shape[1])       loss = loss.sum()
          loss.backward()                                    
  - Feb 20, 2020:
      > add print label when simulation started
      > update label format
      > modify csv file name to include label
      > pass label into make_folder helper function
      > add code for finding date
      > remove double confirmation for deleting data after exception
  - Feb 19, 2020:
      > add label column to initial_configs, which will be saved into the results csv
      > modify config file to include label
  - Feb 4, 2020:
      > add code to save checkpoints, and load from checkpoints
      > modify saving results_df to both results folder and simulation folder to
        just saving one, then copying for the second (to save time)
  - Jan 21, 2020:
      > add anchor dilution and anchor order, as well as random seed to csv file
      > modify filename naming convention
  - Jan 19, 2020:
      > add gzip compression to csv file
  - Jan 18, 2020:
      > revise code to duplicate csv data for optimizer settings
  - Jan 17, 2020:
      > change train function to handle non-user created exceptions
      > change save_freq to plot_freq, and use save_freq to indicate freqeuncy of saving data for csv file
      > add if statements to control frequency of printing statistics, and saving data for csv file
      > add code to create csv file to save results
  - Jan 12, 2020:
      > add parameter in make_plot functions to plot red line when anchors are added
  - Jan 08, 2020:
      > typo: under __init__, self.types = [...] fixed (LFEEXPT -> LFE) as LFEEXPT type does not exist after collapsing
      > dataset filepaths have been replaced with config file inputs
      > modify calculation of loss to do a weighted mean:
            BEFORE:                   AFTER:
            loss = loss*freq          loss = loss*freq 
            loss = loss.mean()        loss = loss.sum()/(freq.sum()*loss.shape[1])
            loss.backward()           loss.backward()
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
import datetime
import numpy as np

import configparser

from .plaut_dataset import plaut_dataset
from .model import plaut_net
from .helpers import *

phoneme_onset = np.array(['s', 'S', 'C', 'z', 'Z', 'j', 'f', 'v', 'T', 'D',
                 'p', 'b', 't', 'd', 'k', 'g', 'm', 'n', 'h', 'l', 'r', 'w', 'y'])
phoneme_vowel = np.array(['a', 'e', 'i', 'o', 'u', '@',
                 '^', 'A', 'E', 'I', 'O', 'U', 'W', 'Y'])
phoneme_codas = np.array(['r', 'l', 'm', 'n', 'N', 'b', 'g', 'd', 'ps', 'ks',
                 'ts', 's', 'z', 'f', 'v', 'p', 'k', 't', 'S', 'Z', 'T', 'D', 'C', 'j'])



class simulator():
    def __init__(self, filename='config.cfg'):
         
        self.cfg_filename = filename

        # Load configuration file
        self.config = configparser.ConfigParser()
        self.config.read(filename)
        
        # Define word types to calculate accuracy for
        self.types = ["HEC", "HRI", "HFE", "LEC", "LFRI", "LFE"] # calculate accuracy of these types
        self.anc_types = ["ANC_REG", "ANC_EXC", "ANC_AMB"]
        self.probe_types = ["PRO_REG", "PRO_EXC", "PRO_AMB"]
        
        # set manual seed
        torch.manual_seed(int(self.config['setup']['random_seed']))
        
    def load_data(self):
        '''
        IMPORTING DATASET
        > import dataset from csv files
        > multiple csv files can be concatenated in format: ["filename1", True/False, "filename2", True/False, ...]
        > True/False indicates whether to override frequencies to log(2)
        '''
        
        plaut_filepath = self.config['dataset']['plaut']
        anc_filepath = self.config['dataset']['anchor']
        probe_filepath = self.config['dataset']['probe']
        anc_freq = self.config['dataset']['anc_freq']
        anc_freq = None if anc_freq == "" else float(anc_freq)
        anc_dilation = self.config['dataset']['anc_dilation']
        anc_dilation = None if anc_dilation == "" else int(anc_dilation)
        
        if anc_freq and anc_dilation:
            anc_freq = np.log(anc_freq + 2) / anc_dilation
        
        # plaut_dataset takes two parameters: filepaths, frequencies
        self.plaut_ds = plaut_dataset([plaut_filepath], [None])
        self.anc_ds = plaut_dataset([anc_filepath], [None])
        self.probe_ds = plaut_dataset([probe_filepath], [None])
        self.plaut_anc_ds = plaut_dataset([plaut_filepath, anc_filepath], [None, anc_freq])
       
        
        '''
        INITIALIZE DATALOADERS
        '''
        self.plaut_loader = DataLoader(self.plaut_ds, batch_size=len(self.plaut_ds), num_workers=0)
        self.anc_loader = DataLoader(self.anc_ds, batch_size=len(self.anc_ds), num_workers=0)
        self.probe_loader = DataLoader(self.probe_ds, batch_size=len(self.probe_ds), num_workers=0)
        self.plaut_anc_loader = DataLoader(self.plaut_anc_ds, batch_size=len(self.plaut_anc_ds), num_workers=0)

    def train(self):
        try: # run training function
            self.load_data()
            self.train_function()
        except: # if interrupted by keyboard
            raise
            if input("An exception occured. Delete plots and checkpoints? [y/n] \n  > ").lower() in ['y', 'yes']:
                shutil.rmtree(self.rootdir)
                print("Simulation results deleted. Error is shown below:")
            else:
                save_notes(self.rootdir)
                print("Simulation results saved. Error is shown below:")
            raise
        else: # no exception
            print("Training Completed!")
            save_notes(self.rootdir)
            print("Simulation results saved.")
            return None
        
        return None
    
    def train_function(self):
        '''
        ================================================
        1/ INITIAL SETUP
        ================================================
        '''
        
        '''
        --------------------------------------------
        1.1/ LOAD CONFIG SETUP
        --------------------------------------------
        '''
        # Load general settings from configuration file
        total_epochs = int(self.config['setup']['total_epochs'])
        anchor_epoch = int(self.config['setup']['anchor_epoch'])
        print_freq = int(self.config['setup']['print_freq'])
        plot_freq = int(self.config['setup']['plot_freq'])
        save_freq = int(self.config['setup']['save_freq'])
        cp_epochs = [int(x) for x in self.config['setup']['checkpoint_epochs'].split(',')]
        cp_name = self.config['setup']['checkpoint_name']
        prev_checkpoint = self.config['setup']['prev_checkpoint']
        label = self.config['setup']['label']
        
        if label == '':
            label = input("Enter label for simulation:")

        if prev_checkpoint != '':
            prev_checkpoint = torch.load(prev_checkpoint)

        # Load optimizer settings from configuration file
        starts, optims, lrates, momenta, wds = [], [], [], [], []
        i = 1
        while True:
            try:
                for x, y in zip([starts, optims, lrates, momenta, wds], ['start', 'optim', 'lr', 'momentum', 'wd']):
                    x.append(self.config['part'+str(i)][y])
                i = i + 1
            except:
                break

        # determine initial configurations
        # dilution: 1 means N, 2 means N/2, 3 means N/3
        # anchor order: 1 means 1->2->3, 3 means 3->2->1
        random_seed = self.config['setup']['random_seed']
        if self.config['dataset']['anc_dilation'] != "":
            dilution = int(self.config['dataset']['anc_dilation'])
        else:
            dilution = self.config['dataset']['anchor'].split('.')[-2][-1]

        anchor_order = 1 if self.config['dataset']['anchor'].split('.')[-2].split('_')[-1][0:-1] == 'new' else 3 
        
        '''
        --------------------------------------------
        1.2/ SAVE INITIAL CONFIG
        --------------------------------------------
        '''
        # calculate total # of samples in dataset
        total_samples = len(self.plaut_ds)+len(self.anc_ds)+len(self.probe_ds)
        
        # date
        now = datetime.datetime.now()
        date = now.strftime("%b").lower()+now.strftime("%d")
        
        # create label for simulation
        label = label+"-S{}D{}O{}-".format(random_seed, dilution, anchor_order)+date
        print("Label for simulation:", label)
        
        if cp_name == '':
            cp_name = label
        
        # save initial configurations
        initial_configs = {
            'dilution': [dilution for j in range(total_samples*total_epochs)],
            'anchor_order': [anchor_order for j in range(total_samples*total_epochs)],
            'random_seed': [random_seed for j in range(total_samples*total_epochs)],
            'label': [label for i in range(total_epochs*total_samples)],
            'anchors_added': [0 for i in range(anchor_epoch*total_samples)] + \
                                           [1 for i in range((total_epochs-anchor_epoch)*total_samples)]
        }
        
        # save optimizer settings to add to results_data after training complete
        for config, key in zip([optims, lrates, momenta, wds], ['optim', 'lr', 'momentum', 'weight_decay']):
            temp = []
            for i in range(len(starts)):
                epoch_start = int(starts[i]) # start of optimizer config
                # end of optimizer config is next item in start, or if no more items, then end is total epochs
                try:
                    epoch_end = int(starts[i+1])
                except:
                    epoch_end = total_epochs
                for k in range(total_samples): # once per every word
                    temp += [config[i] for j in range(epoch_start, epoch_end)] # once per every epoch
            initial_configs[key] = temp
            
        '''
        --------------------------------------------
        1.3/ RESULTS FOLDER SETUP
        --------------------------------------------
        '''
        # Create folder to store results
        self.rootdir = make_folder(date=date, dir_label=label, cfg_filename=self.cfg_filename)
        print("Train Results will be stored in: ", self.rootdir)     
  
        '''
        --------------------------------------------
        1.4/ MODEL INITIALIZATION
        --------------------------------------------
        '''
        # Initialize model
        self.model = plaut_net()

        # define loss function
        criterion = nn.BCELoss(reduction='none') #reduction=none allows scaling by frequency afterwards
        
        # set initial dataloader to not have anchors
        data_loader = self.plaut_loader 

        # Initialize arrays to store epochs, train loss, accuracy
        epochs, losses, times = [], [], []
        acc = [[] for i in self.types]
        anc_acc = [[] for i in self.anc_types]
        probe_acc = [[] for i in self.probe_types]
        
        # Initialize dictionary to save data for csv file
        results_data = {
            'epoch': [],
            'example_id': [],
            'orth': [],
            'phon': [],
            'category': [],
            'correct': [],
            'error': []
        }
        
        # starting epoch
        start_epoch = 0
        
        '''
        --------------------------------------------
        1.5/ LOAD CHECKPOINT (IF NECESSARY)
        --------------------------------------------
        '''
        
        # try to load prev checkpoint if existant
        if prev_checkpoint != '':
            self.model.load_state_dict(prev_checkpoint['model_state_dict'])
            epochs = prev_checkpoint['epochs']
            losses = prev_checkpoint['losses']
            acc = prev_checkpoint['acc']
            anc_acc = prev_checkpoint['anc_acc']
            probe_acc = prev_checkpoint['pro_acc']
            start_epoch = epochs[-1]
            while len(starts) > 0:
                if int(starts[0]) <= start_epoch:
                    optimizer = optims[0]
                    lr = float(lrates[0])
                    m = float(momenta[0])
                    wd = float(wds[0])
                    for i in [starts, optims, lrates, momenta, wds]:
                        i.pop(0)
                else:
                    break
            if optimizer == 'Adam':
                    optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
            elif optimizer == 'SGD':
                optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=m, weight_decay=wd)
            else:
                print("ERROR: Use either Adam or SGD optimizer")
                return None
            optimizer.zero_grad()
        print("Starting Epoch: ", start_epoch)
            
        '''
        ================================================
        2/ TRAINING LOOP
        ================================================
        '''
        # train for specified # of epochs
        for epoch in range(start_epoch, total_epochs): 
            # start timer
            epoch_time = time.time()
            
            # switch optimizer when/if specified in config file
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
                    i.pop(0) # optimizer changed, pop first item off of arrays so change is not repeated
            
            # after specified # of epochs, add anchors
            if epoch == anchor_epoch: 
                data_loader = self.plaut_anc_loader
            
            # initialize avg loss
            avg_loss = 0 
            
            '''
            --------------------------------------------
            2.1/ FWD + BWD PASSES
            --------------------------------------------
            '''
            # iterate through data loader
            for i, data in enumerate(data_loader): 
                # extract frequency, inputs, labels
                freq = data["frequency"].view(-1, 1) # reshape to [batch_size x 1] to match output size
                inputs = data["graphemes"]
                labels = data["phonemes"]

                #forward pass + backward pass + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                loss = loss*freq
                loss = loss.sum() # find *weighted* mean of loss
                
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            '''
            --------------------------------------------
            2.2/ STATISTICS CALCULATION AND SAVING
            --------------------------------------------
            '''
            # save loss and calculate accuracy every X epochs
            if epoch % save_freq == save_freq - 1:
                # calculate loss and save loss to array
                losses.append(avg_loss)
                epochs.append(epoch+1)
                
                for temp_loader, word_types, acc_array in zip([self.plaut_loader, self.anc_loader, self.probe_loader],[self.types, self.anc_types, self.probe_types], [acc, anc_acc, probe_acc]):
                    # calcuate accuracy and append to array
                    temp_acc, temp_data = get_accuracy(self.model, temp_loader, word_types, vowels_only=True)
                    
                    for i in range(len(word_types)):
                        acc_array[i].append(temp_acc[i])
                
                    # update results data for csv
                    results_data['epoch'] = results_data['epoch'] + [epoch+1 for i in range(len(temp_data['example_id']))]
                    for i in ['example_id', 'orth', 'phon', 'category', 'correct', 'error']:
                        results_data[i] = results_data[i] + temp_data[i]
            
            # checkpoint if specified
            if epoch+1 in cp_epochs:
                save_checkpoint(cp_name, self.rootdir, self.model, optimizer, epochs, losses, acc, probe_acc, anc_acc)
                        
            '''
            --------------------------------------------
            2.3/ PLOTTING
            --------------------------------------------
            '''
            # save loss and accuracy plots every X epochs
            if epoch % plot_freq == plot_freq - 1:              
                for ydata, labels, ylabel, title in zip( \
                    [[losses], acc, anc_acc, probe_acc], \
                    [["Train Loss"], self.types, self.anc_types, self.probe_types], \
                    ["Loss", "Accuracy", "Accuracy", "Accuracy"], \
                    ["Training Loss", "Training Accuracy", "Anchor Accuracy", "Probe Accuracy"]):
                    
                    make_plot(epochs, ydata, labels, "Epoch", ylabel, title, anchor=int(self.config['setup']['anchor_epoch']), save=True, filepath=self.rootdir+"/"+title+"/epoch_"+str(epoch+1)+".jpg", show=self.config['setup']['show_plots'])
            '''
            --------------------------------------------
            2.4/ PRINTING OF STATISTICS
            --------------------------------------------
            '''
            # calculate time elapsed over epoch
            time_elapsed = time.time() - epoch_time
            times.append(time_elapsed)
            
            # print stats every X epochs
            if epoch % print_freq == print_freq - 1:
                print("[EPOCH %d] \t loss: %.6f \t time: %.4f \r" % (epoch+1, avg_loss, time_elapsed))
                
        '''
        ================================================
        3/ AFTER TRAINING IS COMPLETED
        ================================================
        '''
        # plot final loss and accuracy line plots and save
        for ydata, labels, ylabel, title in zip(\
            [[losses], acc, anc_acc, probe_acc], \
            [["Train Loss"], self.types, self.anc_types, self.probe_types], \
            ["Loss", "Accuracy", "Accuracy", "Accuracy"], \
            ["Training Loss", "Training Accuracy", "Anchor Accuracy", "Probe Accuracy"]):
                    
            make_plot(epochs, ydata, labels, "Epoch", ylabel, title, anchor=int(self.config['setup']['anchor_epoch']), save=True, filepath=self.rootdir+"/"+title+" Final.jpg", show=self.config['setup']['show_plots'])
        
        # plot final accuracy bar plots and save
        make_bar(self.types, [i[-1] for i in acc], "Category", "Accuracy", "Final Training Accuracy", save=True, filepath=self.rootdir+"/Training Accuracy Bar.jpg", show=self.config['setup']['show_plots'])
        make_bar(self.anc_types, [i[-1] for i in anc_acc], "Category", "Accuracy", "Final Anchor Accuracy", save=True, filepath=self.rootdir+"/Anchor Accuracy Bar.jpg", show=self.config['setup']['show_plots'])
        make_bar(self.probe_types, [i[-1] for i in probe_acc], "Category", "Accuracy", "Final Probe Accuracy", save=True, filepath=self.rootdir+"/Probe Accuracy Bar.jpg", show=self.config['setup']['show_plots'])
        
        # print average time taken
        print("Average Time per epoch: ", sum(times)/len(times))
        
        # create csv file for results
        filename = 'warping-dilation-'+label+'.csv.gz' # extract file name from folder filepath
        results_data.update(initial_configs) # include optimizer settings
        results_df = pd.DataFrame({key:pd.Series(value) for key, value in results_data.items()}) # create dataframe
        results_df.to_csv(self.rootdir+"/"+filename, index=False, compression='gzip') # save as csv in simulation folder
        shutil.copyfile(self.rootdir+"/"+filename, "/"+"/".join(self.rootdir.split('/')[:-1])+"/"+filename) # copy to results folder
        return None
    
    def test(self):
        # Load configuration file
        self.test_config = configparser.ConfigParser()
        self.test_config.read("test_config.cfg")
        label = self.test_config['setup']['label']
        
        if label == '':
            label = input("Enter label for simulation:")
        
        
        # date
        now = datetime.datetime.now()
        date = now.strftime("%b").lower()+now.strftime("%d")
        
        # create label for simulation
        label = label+'-'+date
        print("Label for simulation:", label)
        
        # Create folder to store results
        self.rootdir = make_folder(date=date, dir_label=label)
        print("Test Results will be stored in: ", self.rootdir)     

        # Make a copy of config file in results folder
        shutil.copyfile("test_config.cfg", self.rootdir+"/test_config.cfg")
        
        # (re)-initialize model
        self.model = plaut_net()
        
        # load checkpoint and model parameters
        checkpoint = torch.load(self.test_config['setup']['checkpoint'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # create dataloader for test data
        self.test_ds = plaut_dataset([self.test_config['setup']['test']], [None])
        self.test_loader = DataLoader(self.test_ds, batch_size=len(self.test_ds), num_workers=0)
        
        # test model
        for i, data in enumerate(self.test_loader):
            inputs = data["graphemes"]
            
            outputs = self.model(inputs)
        
        # change outputs to numpy array
        outputs = outputs.detach().numpy()
        
        # save csv file with original outputs
        output_df = pd.DataFrame(data=outputs, columns=['o'+str(i) for i in range(outputs.shape[1])])
        output_df.index.name = 'word_id'
        
        output_df = self.test_ds.df[['word_id','orth']].merge(output_df, on='word_id').set_index('word_id')
        
        output_df.to_csv(self.rootdir+'/model_outputs.csv')
        
        
        # calculate entropy
        entropy = outputs * np.log2(outputs) + (1-outputs)*np.log2(1-outputs) - np.log2(0.5)
        
        entropy_df = pd.DataFrame(data=entropy, columns=['e'+str(i) for i in range(entropy.shape[1])])
        entropy_df.index.name = 'word_id'
        
        entropy_df = self.test_ds.df[['word_id','orth']].merge(entropy_df, on='word_id').set_index('word_id')
        
        entropy_df.to_csv(self.rootdir+'/entropy.csv')
        
        # split output into onset, vowel, coda
        onset = outputs[:, 0:23]
        vowel = outputs[:, 23:37]
        codas = outputs[:, 37:61]
        
        # find most active phoneme, and unit activity
        max_onset = phoneme_onset[onset.argmax(axis=1)]
        max_vowel = phoneme_vowel[vowel.argmax(axis=1)]
        max_codas = phoneme_codas[codas.argmax(axis=1)]
        
        max_onset_activity = onset.max(axis=1)
        max_vowel_activity = vowel.max(axis=1)
        max_codas_activity = codas.max(axis=1)
        
        # sum entropy across different units for onset, vowel, coda
        e_onset = entropy[:, 0:23].sum(axis=1)
        e_vowel = entropy[:, 23:37].sum(axis=1)
        e_codas = entropy[:, 37:61].sum(axis=1)
        
        results_data = {'most_active_onset': max_onset,
                        'most_active_onset_value': max_onset_activity,
                        'entropy_sum_onset': e_onset,
                        'most_active_vowel': max_vowel,
                        'most_active_vowel_value': max_vowel_activity,
                        'entropy_sum_vowel': e_vowel,
                        'most_active_coda': max_codas,
                        'most_active_coda_value': max_codas_activity,
                        'entropy_sum_coda': e_codas}
        
        results_df = pd.DataFrame(data=results_data)
        results_df.index.name = 'word_id'
        
        results_df = self.test_ds.df[['word_id', 'orth']].merge(results_df, on='word_id').set_index('word_id')
        
        results_df.to_csv(self.rootdir+'/results.csv')
