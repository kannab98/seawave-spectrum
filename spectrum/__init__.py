
import logging
import sys, os
import toml
from numpy import allclose
from . import *



__all__ = ["spectrum", "omega", "k", "dopler"]










logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')



sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)

logger.debug('Welcome to project repo: https://github.com/kannab98/seawavepy')



        
        
name = 'config.toml'

cwd = os.getcwd()

cfg = os.path.join(cwd, name)
if os.path.isfile(cfg):
    configfile = cfg
else:
    configfile = os.path.join(os.path.dirname(__file__), name)

config = toml.load(configfile)
logger.info('Load config from %s' % configfile)

fh = logging.FileHandler('modeling.log')
fh.setFormatter(formatter)
logger.addHandler(fh)



def exit_handler(srf):
    logger.info(srf)
    for coord in ['X', 'Y']:
        if allclose(srf[coord].values[0,:,:], srf[coord].values):
            srf[coord] = (["x", "y"], srf[coord].values[0,:,:]) 

    if config["Dataset"]["File"]:
        logger.info("Save file to %s" % os.path.join(os.getcwd(), config["Dataset"]["File"]))
        srf.to_netcdf(config['Dataset']['File'])





from .core import *
from .dispersion import *
spectrum = spectrum()

