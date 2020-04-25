'''
helpers.py

Description: Helper Functions for Plaut Model
  - get_accuracy: to calculate the accuracy of a set of given word types in a dataset
  - make_plot: makes a line plot with multiple lines
  - make_folder: makes a new folder to store simulation results
  - save_notes: function to save notes regarding simulation results at end of simulation

Date Created: November 27, 2019

Revisions:
  - Mar 30, 2020:
      > add save_checkpoint function (moved from simulator.py
  - Feb 20, 2020:
      > modify make_folder to allow custom name based on label
      > move code for finding date to simulator.py
  - Jan 17, 2020:
      > modify accuracy calculation to simply be based on highest activity
      > modify get_accuracy function to also return dictionary of items needed to save csv
  - Jan 12, 2020:
      > add gridlines on plot
      > add line to indicate when anchors are added
  - Jan 08, 2020:
      > set bounds on y-axis to be [-0.05, 1.05] for accuracy plots
      > reduce saved plot dpi from 200 to 150
  - Jan 05, 2020: multiple revisions, see below:
      > remove .float when loading data from dataloader, since this is now done when creating the dataset
      > add save_notes function to add a notes.txt file after running simulation to save any notes
      > add make_bar function to create bar graphs for accuracy at end of simulation
      > add show_plot parameter in make_plot and make_bar to control whether to print plots
  - Jan 04, 2020: modify make_folder to create subdirectories for plots
  - Jan 01, 2020: multiple revisions, see below:
      > migrate code to make new folder (make_folder) for simulation from plaut_model.ipynb
  - Nov 28, 2019: modify get_accuracy to have option to analyze vowels only
  - Nov 27, 2019: multiple changes, see below:
      > create file, copy get_accuracy from plaut_model.ipynb file
      > modify get_accuracy to take a list of word types at once
      > write make_plot function

'''

import pandas as pd
import torch
from matplotlib import pyplot as plt
import os
from pathlib import Path
import shutil

# function to get the accuracy of a particular category
def get_accuracy(model, data_loader, cat=['All'], vowels_only=False):
    correct = [0.0 for i in cat]
    total = [0.0 for i in cat]
    accuracy = [0.0 for i in cat]
    
    for i, data in enumerate(data_loader):  # get batch from dataloader
        
        # extract inputs, labels, type from batch
        inputs = data["graphemes"]
        labels = data["phonemes"]
        types = pd.DataFrame(data["type"])
        
        # find prediction using model, then round to 0 or 1
        outputs = model(inputs)  
       
        # The following section compares the outputs with the labels, torch.eq performs an element-wise
        # comparison between the two input vectors, and .sum sums up the amount of indentical (i.e. correct)
        # elements in each word -> compare is dim [# of samples]
        
        if vowels_only == True:
            compare = torch.eq(outputs[:, 23:37].argmax(dim=1), labels[:, 23:37].argmax(dim=1))
            #compare = torch.eq(outputs[:, 23:37], labels[:, 23:37]).sum(dim=1) # compare vowel section only with labels
            #compare_len = 14 # length to compare is only # of vowels
        else:
            raise Exception("Code needs to be updated for NOT vowels only case")
            compare = torch.eq(outputs, labels).sum(dim=1)  # compare with labels
            compare_len = 61 # length to compare is entire phonology vector
        
        for j in range(len(cat)): # for each desired type
            if cat[j] == 'All':
                correct[j] = compare.sum()[0]
                #correct[j] += torch.eq(compare, compare_len).sum().item() # count as correct if all desired elements match label
                #total[j] += len(compare) # accumulate number of samples
            else:
                curr_type = types.apply(lambda x: x == cat[j])  # check for desired type
                #temp_compare = pd.DataFrame(compare) # create dataframe using compare to faciliate next line (vector comparison)
                temp_compare = pd.DataFrame(compare)
                correct[j] += ((curr_type == True) & (temp_compare == True)).sum()[0] # correct if desired type AND all desired elements match label
                total[j] += (curr_type == True).sum()[0] # count all of the desired type
                
    
    # calculate accuracy: divide correct samples by total samples
    for i in range(len(accuracy)):
        accuracy[i] = correct[i]/total[i] 
        
    temp_data = {
        'correct': [int(i) for i in compare.tolist()],
        'orth': data['orth'],
        'phon': data['phon'],
        'category': data['type'],
        'example_id': [i+1 for i in range(len(compare))],
        'error': [0 for i in range(len(compare))]
    }
    
    return accuracy, temp_data

def make_plot(x_data, y_data, labels, x_label, y_label, title, anchor=500, save=False, filepath=None, show=True):
    # initialize figure
    plt.figure()  

    # for multiple lines on same graph, plot one at a time
    for data, label in zip(y_data, labels):
        plt.plot(x_data, data, label=label)
    
    if max(x_data) >= anchor:
        plt.plot([anchor, anchor], [-1, 2], lw=0.5, color='red', label="Add Anchors")
    
    # set xlim and yclim
    plt.xlim(0-max(x_data)*0.05, max(x_data)*1.05)
    if "Accuracy" in title:
        plt.ylim(-0.05, 1.05)
    else:
        plt.ylim(0-max(y_data[0])*0.05, max(y_data[0])*1.05)

    #axis labels, title, legend, gridlines
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(b=True, which='both', axis='both')
    plt.title(title)
    plt.legend(loc='best')
    
    # save plot at filepath if needed
    if save == True:
        plt.savefig(filepath, dpi=150)
    
    # show plot in Jupyter Notebook cell output
    if show == True:
        plt.show()

    return None

def make_bar(x_data, y_data, x_label, y_label, title, save=False, filepath=None, show=True):
    # initialize figure
    plt.figure()  

    # plot bar graph
    plt.bar(x_data, y_data)
    
    # set ylim for accuracy plots only
    if "Accuracy" in title:
        plt.ylim(-0.05, 1.05)
    
    #axis labels, title, legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
            
    # save plot at filepath if needed
    if save == True:
        plt.savefig(filepath, dpi=150)
        
    # show plot in Jupyter Notebook cell output
    if show == True:
        plt.show()
    
    return None

def make_folder(date, dir_label=None, cfg_filename="config.cfg"):
    # create a new folder for every run
    # path = Path(os.getcwd()).parent #get parent (Plaut_Model) directory filepath
    path = Path(os.getcwd())

    if dir_label == None: # if no directory label specified
        i = 1
        while True:
            try:
                rootdir = str(path)+"/results/"+date+"_test"+'{:02d}'.format(i)
                os.mkdir(rootdir)
                for subdir in ["Training Loss", "Training Accuracy", "Anchor Accuracy", "Probe Accuracy"]:
                    os.mkdir(rootdir+"/"+subdir)
                break
            except:
                i += 1
    else: # if name specified
        rootdir = str(path)+"/results/"+dir_label
        os.mkdir(rootdir)
        for subdir in ["Training Loss", "Training Accuracy", "Anchor Accuracy", "Probe Accuracy"]:
            os.mkdir(rootdir+"/"+subdir)

    # Make a copy of config file in results folder
    shutil.copyfile(cfg_filename, rootdir+"/config.cfg")

    return rootdir

def save_notes(rootdir):
    with open(rootdir+"/notes.txt", 'w') as writer: # make notes file inside results folder
        writer.write("NOTES: \n")
        print("Add any notes below to save with results:")
        
        # write each note on new line, until nothing is entered
        notes = input(" > ") 
        while notes != "":
            writer.write(" > "+notes+"\n")
            notes = input(" > ")
    return None

def save_checkpoint(cp_name, rootdir, model, optimizer, epochs, losses, acc, pro_acc, anc_acc):
    # extract checkpoint directory
    cp_dir = rootdir[0:rootdir.index('results')]+'checkpoints/'
    
    torch.save({
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'acc': acc,
        'pro_acc': pro_acc,
        'anc_acc': anc_acc
    }, cp_dir+cp_name+'_epoch_'+str(epochs[-1])+'.tar')
    
    return None
