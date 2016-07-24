from __future__ import division
import os
import glob

import numpy as np
import h5py
import scipy
from scipy.io import loadmat, savemat
from optparse import OptionParser
from tqdm import tqdm

import .signal_processing.HTK_hilb as htk
import .signal_processing.downsample as dse
import .signal_processing.subtract_CAR as scar
import .signal_processing.apply_linenoise_notch as notch
import .signal_processing.apply_hilbert_transform as aht
import .signal_processing.delete_bad_time_segments as dbts

import pdb

__authors__ = "Alex Bujan (adapted from Kris Bouchard)"


def load_electrode_labels(subj_dir):
    """
    Get anatomy. Try newest format, and then progressively earlier formats.

    :param subj_dir:
    :return: np.array of labels as strings
    """
    anatomy_filename = glob.glob(os.path.join(subj_dir, '*_anat.mat'))
    elect_labels_filename = glob.glob(os.path.join(subj_dir, 'elec_labels.mat'))
    hd_grid_file = glob.glob(os.path.join(subj_dir, 'Imaging', 'elecs', 'hd_grid.mat'))
    TDT_elecs_file = glob.glob(os.path.join(subj_dir, 'Imaging', 'elecs', 'TDT_elecs_all.mat'))

    if TDT_elecs_file:
        mat_in = loadmat(TDT_elecs_file[0])
        try:
            electrode_labels = mat_in['anatomy'][:, -1].ravel()
            print('anatomy from TDT_elecs_all')
            return np.array([a[0] for a in electrode_labels])
        except:
            pass

    if hd_grid_file:
        mat_in = loadmat(hd_grid_file[0])
        if 'anatomy' in mat_in:
            electrode_labels = np.concatenate(mat_in['anatomy'].ravel())
            print('anatomy hd_grid')

            return electrode_labels

    if anatomy_filename:
        anatomy = loadmat(anatomy_filename[0])
        anat = anatomy['anatomy']
        electrode_labels = np.array([''] * 256, dtype='S6')
        for name in anat.dtype.names:
            electrode_labels[anat[name][0, 0].flatten() - 1] = name
        electrode_labels = np.array([word.decode("utf-8") for word in electrode_labels])
        # electrode_labels = np.array([item[0][0] if len(item[0]) else '' for item in anatomy['electrodes'][0]])

    elif elect_labels_filename:
        a = scipy.io.loadmat(elect_labels_filename[0])
        electrode_labels = np.array([elem[0] for elem in a['labels'][0]])

    else:
        electrode_labels = None
        print('No electrode labels found')

    return electrode_labels


def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    parser.add_option("--subject",type="string",default='GP31',\
        help="Subject code")

    parser.add_option("--block",type="string",default='B1',\
        help="Block number eg: 'B1'")

    parser.add_option("--path",type="string",default='',\
        help="Path to the data")

    parser.add_option("--rate",type="float",default=400.,\
        help="Sampling rate of the processed signal (optional)")

    parser.add_option("--vsmc",action='store_true',\
        dest='vsmc',help="Include vSMC electrodes only (optional)")

    parser.add_option("--store",action='store_true',\
        dest='store',help="Store results (optional)")

    parser.add_option("--ct",type="float",default=87.75,\
        help="Center frequency of the Gaussian filter (optional)")

    parser.add_option("--sd",type="float",default=3.65,\
        help="Standard deviation of the Gaussian filter (optional)")

    parser.add_option("--srf",type="float",default=1e4,\
        help="Sampling rate factor. Read notes in HTK.py (optional)")

    (options, args) = parser.parse_args()

    assert options.path!='',IOError('Inroduce a correct data path!')

    if options.vsmc:
        vsmc=True
    else:
        vsmc=False

    if options.store:
        store=True
    else:
        store=False

    transform(path=options.path,subject=options.subject,block=options.block,\
              rate=options.rate,vsmc=vsmc,ct=options.ct,sd=options.sd,\
              store=store,srf=options.srf)


def transform(blockpath, rate=400., vsmc=False, cts=None, sds=None, srf=1e4, suffix=''):
    """
    Takes raw LFP data and does the standard hilb algorithm:
    1) CAR
    2) notch filters
    3) Hilbert transform on different Gaussian bands
    ...

    Saves to os.path.join(blockpath, subject + '_B' + block + '_Hilb.h5')

    Parameters
    ----------
    blockpath
    rate
    vsmc
    cts: filter center frequencies. If None, use Chang lab defaults
    sds: filer standard deviations. If None, use Chang lab defaults
    srf: htk multiple

    takes about 20 minutes to run on 1 10-min block

    Returns
    -------
    Nothing. The result is too big to have in memory and is saved filter-by-filter to a h5.



    #ct=87.75,sd=3.65
    """

    ######## setup bandpass hilbert filter parameters
    if cts is None:
        fq_min = 4.0749286538265
        fq_max = 200.
        scale = 7.
        cts = 2 ** (np.arange(np.log2(fq_min) * scale, np.log2(fq_max) * scale) / scale)
    else:
        cts = np.array(cts)

    if sds is None:
        sds = 10 ** ( np.log10(.39) + .5 * (np.log10(cts)))
    ########
    #pdb.set_trace()


    s_path, blockname = os.path.split(blockpath)

    #b_path = '%s/%s/%s_%s'%(path,subject,subject,block)

    # first, look for downsampled ECoG in block folder
    ds_ecog_path = os.path.join(blockpath, 'ecog400', 'ecog.mat')
    if os.path.isfile(ds_ecog_path):
        print('loading ecog')
        with h5py.File(ds_ecog_path, 'r') as f:
            X = f['ecogDS']['data'][:].T
            fs = f['ecogDS']['sampFreq'][0]
    else:
        """
        Load raw HTK files
        """
        rd_path = os.path.join(blockpath, 'RawHTK')
        HTKoutR = htk.read_HTKs(rd_path)
        X = HTKoutR['data']

        """
        Downsample to 400 Hz
        """
        X = dse.downsample_ecog(X, rate, HTKoutR['sampling_rate'] / srf)

        os.mkdir(os.path.join(blockpath, 'ecog400'))
        savemat(ds_ecog_path, {'ecogDS':{'data': X, 'sampFreq': rate}})

    """
    Subtract CAR
    """
    X = scar.subtractCAR(X)

    """
    Select electrodes
    """
    #electrodes = loadmat('%s/%s/Anatomy/%s_anatomy.mat'%(path,subject,subject))
    labels = load_electrode_labels(s_path)
    if vsmc:
        elects = np.where((labels == 'precentral') | (labels == 'postcentral'))[0]
    else:
        elects = range(256)

    badElects = np.loadtxt('/%s/Artifacts/badChannels.txt'%blockpath)-1
    # I would prefer to keep track of bad channels with NaNs. If you remove them, it becomes cumbersome to keep
    # track of the shift in electrode numbers. I would also be open to masked arrays.
    #elects = np.setdiff1d(elects,badElects)
    X[badElects.astype('int')] = np.nan


    X = X[elects]

    """
    Discard bad segments
    """
    #TODO
#    badSgm = loadmat('%s/Artifacts/badTimeSegments.mat'%b_path)['badTimeSegments']
#    dbts.deleteBadTimeSegments(X,rate,badSgm)

    """
    Apply Notch filters
    """
    X = notch.apply_linenoise_notch(X, rate)

    """
    Apply Hilbert transform and store
    """

    hilb_path = os.path.join(blockpath, blockname + '_Hilb' + suffix +'.h5')
    with h5py.File(hilb_path, 'w') as f:
        dset_real = f.create_dataset('X_real', (len(cts), X.shape[0], X.shape[1]), 'float32', compression="gzip")
        dset_imag = f.create_dataset('X_imag', (len(cts), X.shape[0], X.shape[1]), 'float32', compression="gzip")
        for i, (ct, sd) in enumerate(tqdm(zip(cts, sds), 'applying Hilbert transform', total=len(cts))):
            dat = aht.apply_hilbert_transform(X, rate, ct, sd)
            dset_real[i] = dat.real.astype('float32')
            dset_imag[i] = dat.imag.astype('float32')

        for dset in (dset_real, dset_imag):

            dset.dims[0].label = 'filter'
            for val, name in ((cts, 'filter_center'), (sds, 'filter_sigma')):
                if name not in f.keys():
                    f[name] = val
                dset.dims.create_scale(f[name], name)
                dset.dims[0].attach_scale(f[name])

            dset.dims[1].label = 'channel'
            dset.dims[2].label = 'time'
        f.attrs['sampling_rate'] = rate



if __name__=='__main__':
    main()
