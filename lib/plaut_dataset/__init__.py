import os
cwd = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, os.path.join(cwd, '../..'))

from code.plaut_dataset import plaut_dataset as plaut_dataset_brian

class plaut_dataset(plaut_dataset_brian):
    pass

