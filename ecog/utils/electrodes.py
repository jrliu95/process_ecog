from __future__ import division
import glob, h5py, os

import numpy as np
from scipy.io import loadmat


__all__ = ['load_anatomy', 'load_bad_electrodes']


def load_anatomy(nwb):
    electrode_labels = nwb.ec_electrodes['location'].data[:]

    return electrode_labels


def load_bad_electrodes(nwb):
    """
    Load bad electrodes.

    Parameters
    ----------
    nwb : NWBFile
        Open NWB file for the block.

    Returns
    -------
    bad_electrodes : ndarray
        Python (0-based) indices of bad electrodes.
    """
    try:
        bad_electrodes = nwb.ec_electrodes['bad'].data[:]
    except:
        print('Need to include bad electrodes in the saved NWB files?')
        bad_electrodes = np.array([])

    return bad_electrodes
