"""This scripts contain a few wrapper functions to retrieve TODs based
on names"""

import moby2
from moby2.scripting import get_filebase
from sklearn import preprocessing

fb = get_filebase()

def get_filenames(tod_list):
    """Get the filenames from TOD names
    :param:
        tod_list: list of tod names
    :ret:
        list of tod filenames"""
    return [fb.filename_from_name(tod, single=True) for tod in tod_list]
    
def get_tod(tod_filename):
    """Get tod data from a filename (for internal use)
    :param: 
        tod_filename: a tod filename
    :ret: 
        tod data"""
    return moby2.scripting.get_tod({'filename':tod_filename, 'repair_pointing':True})

def get_tod_data(tod_filename, downsample=0):
    """Get tod data based on filename with mean removal and downsample
    :param:
        tod_filename: a tod filename
        downsample: factor to downsample, 0 means no downsampling
    :ret:
        tod data"""
    _tod = get_tod(tod_filename)  # get raw data
    moby2.tod.remove_mean(_tod)  # remove mean
    if downsample == 0:
        return _tod.data
    else:
        return _tod.data[:,::downsample]

def get_tod_data_list(tod_list, downsample=0):
    """Get tod data based on a list of filenames
    :param:
        tod_list: a list of tod name (not filename)
        downsample: factor to downsample, 0 means no downsampling
    :ret:
        list of tod data"""
    _names = get_filenames(tod_list)  # get filenames for the list
    _data_list = [normalize(get_tod_data(_name, downsample)) for _name in _names]
    return _data_list

def normalize(tod_data):
    """A utility function that normalize tod data
    :param:
       tod_data: tod data
    :ret:
       tod_data"""
    return preprocessing.normalize(tod_data, norm='max')   
