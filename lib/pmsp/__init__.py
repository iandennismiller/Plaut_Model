import os
cwd = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, os.path.join(cwd, '../..'))

from code.simulator import simulator as simulator_brian
from helpers import *

import configparser
import torch

class simulator(simulator_brian):
    def __init__(self, filename='config.cfg'):
         
        # Load configuration file
        self.config = configparser.ConfigParser()
        self.config.read(filename)
        
        print(self.config)

        # Define word types to calculate accuracy for
        self.types = ["HEC", "HRI", "HFE", "LEC", "LFRI", "LFE"] # calculate accuracy of these types
        self.anc_types = ["ANC_REG", "ANC_EXC", "ANC_AMB"]
        self.probe_types = ["PRO_REG", "PRO_EXC", "PRO_AMB"]
        
        # set manual seed
        torch.manual_seed(int(self.config['setup']['random_seed']))
        
