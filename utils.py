import os
import logging

def get_logger(path):
    """Get logger for logging

    Parameters
    ----------
    path : str
        path to log file

    Returns
    -------
    log : logging.Logger
        logger object
    """    
    if not os.path.exists(os.path.dirname(path)):
        print("Creating log directory {}".format(os.path.dirname(path)))
        os.makedirs(os.path.dirname(path))
    if not os.path.exists(path):
        print("Creating log file{}".format(path))
        open(path, 'a').close()

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s", 
                                  datefmt="%a %b %d %H:%M:%S %Y")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log