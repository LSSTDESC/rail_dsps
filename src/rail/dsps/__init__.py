import os
try:
    os.environ['SPS_HOME']
except KeyError:
    os.makedirs(os.path.join(os.getcwd(), 'data'), exist_ok=True)
    with open(os.path.join(os.getcwd(), 'data/emlines_info.dat'),'w') as f:
        f.write('923.148,Ly 923')
    os.environ['SPS_HOME'] = os.getcwd()
from ._version import __version__

from rail.creation.engines.dsps_photometry_creator import *
from rail.creation.engines.dsps_sed_modeler import *