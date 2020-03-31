'''
plaut_dataset.py

Description: Class for custom dataset for Plaut Model

Date Created: November 27, 2019

Revisions:
  - Mar 30, 2020:
      > In get_graphemes: fix error that would occur if word started with 'Y'
      > Before: words that start with Y would have the Y be a vowel
      > After: Y is now placed as onset
  - Mar 03, 2020:
      > modify __init__ in plaut_dataset to allow modification of frequency
      > separate filepaths from frequencies in parameter passing
  - Jan 02, 2019: convert datatype to float here, instead of during training
  - Nov 28, 2019: modify plaut_dataset to add option to "zero" frequencies -> i.e. set to log(2)
  - Nov 27, 2019: create file, copy original class from plaut_model.ipynb file
      > copy get_graphemes/phonemes functions from plaut_model.ipynb
      > modify class to take in filepath instead of dataframe
      > add capability for function to merge multiple csvs into one dataframe

'''

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Mappings for Graphemes
grapheme_onset = ['Y', 'S', 'P', 'T', 'K', 'Q', 'C', 'B', 'D', 'G', 'F', 'V', 'J', 'Z',
                  'L', 'M', 'N', 'R', 'W', 'H', 'CH', 'GH', 'GN', 'PH', 'PS', 'RH', 'SH', 'TH', 'TS', 'WH']
grapheme_vowel = ['E', 'I', 'O', 'U', 'A', 'Y', 'AI', 'AU', 'AW', 'AY', 'EA', 'EE', 'EI',
                  'EU', 'EW', 'EY', 'IE', 'OA', 'OE', 'OI', 'OO', 'OU', 'OW', 'OY', 'UE', 'UI', 'UY']
grapheme_codas = ['H', 'R', 'L', 'M', 'N', 'B', 'D', 'G', 'C', 'X', 'F', 'V', 'J', 'S', 'Z', 'P', 'T', 'K', 'Q', 'BB', 'CH', 'CK', 'DD', 'DG',
                  'FF', 'GG', 'GH', 'GN', 'KS', 'LL', 'NG', 'NN', 'PH', 'PP', 'PS', 'RR', 'SH', 'SL', 'SS', 'TCH', 'TH', 'TS', 'TT', 'ZZ', 'U', 'E', 'ES', 'ED']


# Mappings for Phonemes
phoneme_onset = ['s', 'S', 'C', 'z', 'Z', 'j', 'f', 'v', 'T', 'D',
                 'p', 'b', 't', 'd', 'k', 'g', 'm', 'n', 'h', 'l', 'r', 'w', 'y']
phoneme_vowel = ['a', 'e', 'i', 'o', 'u', '@',
                 '^', 'A', 'E', 'I', 'O', 'U', 'W', 'Y']
phoneme_codas = ['r', 'l', 'm', 'n', 'N', 'b', 'g', 'd', 'ps', 'ks',
                 'ts', 's', 'z', 'f', 'v', 'p', 'k', 't', 'S', 'Z', 'T', 'D', 'C', 'j']


class plaut_dataset(Dataset):
    def __init__(self, filepaths, frequencies):
        # initialize dataframe from csv file
        self.df = pd.DataFrame()
        for file, freq in zip(filepaths, frequencies):
            temp_df = pd.read_csv(file, na_filter=False)
            
            if freq != None:
                temp_df["log_freq"] = freq
                
            self.df = pd.concat([self.df, temp_df])
            
        self.df = self.df.reset_index(drop=True) # reset the index
        
        # get grapheme and phoneme vectors from orthography and phonology
        self.df["graphemes"] = self.df["orth"].apply(lambda x: get_graphemes(x))
        self.df["phonemes"] = self.df["phon"].apply(lambda x: get_phonemes(x))
        

    def __len__(self):
        # return the number of samples in dataframe
        return len(self.df)

    def __getitem__(self, index):
        # return a data sample
        if torch.is_tensor(index):
            index = index.tolist() # convert tensor of indices to list

        return {"type": self.df.loc[index, "type"], # type of word
                "orth": self.df.loc[index, "orth"], # orthography (e.g. ace)
                "phon": self.df.loc[index, "phon"], # phonography (e.g. /As/)
                "frequency": torch.tensor(self.df.loc[index, "log_freq"], dtype=torch.float), # the frequency AFTER log transform
                "graphemes": torch.tensor(self.df.loc[index, 'graphemes'], dtype=torch.float), # vector of graphemes representing orthography
                "phonemes": torch.tensor(self.df.loc[index, 'phonemes'], dtype=torch.float) }# vector of phonemes representing phonology

    
def get_graphemes(word):
    word = str(word).upper() # convert all text to capitals first
    if word == "NAN": # the word null automatically gets imported as "NaN" in dataframe, so fix that
        word = "NULL"
    
    # initialize vectors to zero
    onset = [0 for i in range(len(grapheme_onset))]
    vowel = [0 for i in range(len(grapheme_vowel))]
    codas = [0 for i in range(len(grapheme_codas))]
    
    # for onset: essentially "turn on" corresponding slots for onsets until a vowel is reached
    for i in range(len(word)):
        if word[i] in grapheme_vowel: # vowel found, move on
            if not (i == 0 and word[i] == 'Y'):
                break
        if word[i] in grapheme_onset: # single-letter grapheme found
            onset[grapheme_onset.index(word[i])] = 1
        if word[i:i+2] in grapheme_onset: # double-letter grapheme found
            onset[grapheme_onset.index(word[i:i+2])] = 1
            
    # for vowels
    vowel[grapheme_vowel.index(word[i])] = 1
    if i + 1 < len(word): # check for double-vowel
        if word[i+1] in grapheme_vowel:
            vowel[grapheme_vowel.index(word[i+1])] = 1
        if word[i:i+2] in grapheme_vowel:
            vowel[grapheme_vowel.index(word[i:i+2])] = 1
        # if double-letter vowel found, increment i one more time
        if word[i+1] in grapheme_vowel or word[i:i+2] in grapheme_vowel:
            i += 1
            
    # for codas
    for j in range(i+1, len(word)):
        if word[j] in grapheme_codas: # check for single-letter coda
            codas[grapheme_codas.index(word[j])] = 1
        if word[j:j+2] in grapheme_codas: # check for double-letter coda
            codas[grapheme_codas.index(word[j:j+2])] = 1
        if word[j:j+3] in grapheme_codas: # check for triple-letter coda
            codas[grapheme_codas.index(word[j:j+3])] = 1
            
    # combine and return
    return onset + vowel + codas

# similar idea to graphemes; refer to above for comments
def get_phonemes(phon):
    phon = phon[1:-1]
    onset = [0 for i in range(len(phoneme_onset))]
    vowel = [0 for i in range(len(phoneme_vowel))]
    codas = [0 for i in range(len(phoneme_codas))]

    for i in range(len(phon)):
        if phon[i] in phoneme_vowel:
            break
        if phon[i] in phoneme_onset:
            onset[phoneme_onset.index(phon[i])] = 1

    for j in range(i, len(phon)):
        if phon[j] in phoneme_codas:
            break
        if phon[j] in phoneme_vowel:
            vowel[phoneme_vowel.index(phon[j])] = 1

    for k in range(j, len(phon)):
        if phon[k] in phoneme_codas:
            codas[phoneme_codas.index(phon[k])] = 1
        if phon[k:k+2] in phoneme_codas:
            codas[phoneme_codas.index(phon[k:k+2])] = 1

    return onset + vowel + codas
