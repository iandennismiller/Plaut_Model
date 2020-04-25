import os
cwd = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, os.path.join(cwd, '../..'))

from code.model import plaut_net as plaut_net_brian

class plaut_net(plaut_net_brian):
    pass
