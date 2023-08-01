import os
try:
    os.environ['SPS_HOME']
except KeyError:
    os.environ['SPS_HOME'] = '../examples_data/creation_data/data'
from ._version import __version__

from rail.creation.engines.dsps_photometry_creator import *
from rail.creation.engines.dsps_sed_modeler import *