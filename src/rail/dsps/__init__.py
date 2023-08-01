import os
try:
    os.environ['SPS_HOME']
except KeyError:
    os.makedirs(os.path.join(os.getcwd(), 'data'))
    with open(os.path.join(os.getcwd(), 'data/emlines_info.dat'),'w') as f:
        f.write('test')
    os.environ['SPS_HOME'] = os.path.join(os.getcwd(), 'data')
from ._version import __version__

from rail.creation.engines.dsps_photometry_creator import *
from rail.creation.engines.dsps_sed_modeler import *