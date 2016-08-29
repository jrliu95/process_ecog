from __future__ import division
import os
import glob

import numpy as np
import h5py
import scipy
from scipy.io import savemat
import argparse
try:
    from tqdm import tqdm
except:
    tqdm = lambda x, *args, **kwargs: x

import ecog.signal_processing.HTK_hilb as htk
import ecog.signal_processing.downsample as dse
import ecog.signal_processing.subtract_CAR as scar
import ecog.signal_processing.apply_linenoise_notch as notch
import ecog.signal_processing.apply_hilbert_transform as aht
from ecog.signal_processing.apply_hilbert_transform import gaussian, hamming
import ecog.signal_processing.delete_bad_time_segments as dbts
from ecog.utils import load_electrode_labels, load_bad_electrodes


__authors__ = "Alex Bujan (adapted from Kris Bouchard)"


def main():
    usage = '%prog [options]'

    parser = argparse.ArgumentParser(description='Preprocessing ecog data.')
    parser.add_argument('subject', type=str, help="Subject code")
    parser.add_argument('blocks', type=int, nargs='+',
                        help="Block number eg: '1'")
    parser.add_argument('path', type=str, help="Path to the data")
    parser.add_argument('r', '--rate', type=float, default=200.,
        help="Sampling rate of the processed signal (optional)")
    parser.add_argument('--vsmc', type=bool, default=True,
                        help="Include vSMC electrodes only (optional)")
    parser.add_argument('--ct', type='float', default=None,
                        help="Center frequency of the Gaussian filter (optional)")
    parser.add_argument('--sd', type=float, default=None,
                        help="Standard deviation of the Gaussian filter (optional)")
    parser.add_argument('n', '--neuro', type=bool, default=False,
                        help="Use standard neuroscience boxcar frequency bands")
    parser.add_argument('--srf', type=float, default=1e4,
                        help="Sampling rate factor. Read notes in HTK.py (optional)")
    args = parser.parse_args()

    for block in args.blocks:
        block_path = os.path.join(args.path, args.subject,
                '{}_{}'.format(subject, block))
        transform(block_path, rate=args.rate, vsmc=args.vsmc,
                  ct=args.ct, sd=args.sd,
                  neuro=args.neuro, srf=args.srf)


def transform(block_path, rate=400., vsmc=False, cts=None, sds=None, srf=1e4,
              neuro=False, suffix=''):
    """
    Takes raw LFP data and does the standard hilb algorithm:
    1) CAR
    2) notch filters
    3) Hilbert transform on different bands
    ...

    Saves to os.path.join(block_path, subject + '_B' + block + '_Hilb.h5')

    Parameters
    ----------
    block_path
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

    if neuro:
        bands = ['theta', 'alpha', 'beta', 'high beta', 'gamma', 'high gamma']
        min_freqs = [4, 9, 15, 21, 30, 85]
        max_freqs = [8, 14, 20, 29, 59, 175]
    else:
        if cts is None:
            fq_min = 4.0749286538265
            fq_max = 200.
            scale = 7.
            cts = 2 ** (np.arange(np.log2(fq_min) * scale, np.log2(fq_max) * scale) / scale)
        else:
            cts = np.array(cts)

        if sds is None:
            sds = 10 ** ( np.log10(.39) + .5 * (np.log10(cts)))
        else:
            sds = np.array(sds)


    subj_path, blockname = os.path.split(block_path)

    # first, look for downsampled ECoG in block folder
    ds_ecog_path = os.path.join(block_path, 'ecog400', 'ecog.mat')
    if os.path.isfile(ds_ecog_path):
        print('loading ecog')
        with h5py.File(ds_ecog_path, 'r') as f:
            X = f['ecogDS']['data'][:].T
            fs = f['ecogDS']['sampFreq'][0]
    else:
        """
        Load raw HTK files
        """
        rd_path = os.path.join(block_path, 'RawHTK')
        HTKoutR = htk.read_HTKs(rd_path)
        X = HTKoutR['data']

        """
        Downsample to 400 Hz
        """
        X = dse.downsample_ecog(X, rate, HTKoutR['sampling_rate'] / srf)

        os.mkdir(os.path.join(block_path, 'ecog400'))
        savemat(ds_ecog_path, {'ecogDS':{'data': X, 'sampFreq': rate}})

    """
    Subtract CAR
    """
    X = scar.subtract_CAR(X)

    """
    Select electrodes
    """
    labels = load_electrode_labels(subj_path)
    if vsmc:
        elects = np.where((labels == 'precentral') | (labels == 'postcentral'))[0]
    else:
        elects = range(256)

    bad_elects = load_bad_electrodes(subj_path)
    X[bad_elects] = np.nan

    X = X[elects]

    """
    Apply Notch filters
    """
    X = notch.apply_linenoise_notch(X, rate)

    """
    Apply Hilbert transform and store
    """

    if neuro:
        hilb_path = os.path.join(block_path, blockname + '_neuro_Hilb' + suffix +'.h5')
    else:
        hilb_path = os.path.join(block_path, blockname + '_Hilb' + suffix +'.h5')
    with h5py.File(hilb_path, 'w') as f:
        if neuro:
            dset_real = f.create_dataset('X_real', (len(bands), X.shape[0], X.shape[1]),
                                         'float32', compression="gzip")
            dset_imag = f.create_dataset('X_imag', (len(bands), X.shape[0], X.shape[1]),
                                         'float32', compression="gzip")
            for ii, (min_f, max_f) in enumerate(tqdm(zip(min_freqs, max_freqs),
                                                'applying Hilbert transform',
                                                total=len(batch))):
                kernel = hamming(X, rate, min_f, max_f)
                dat = aht.apply_hilbert_transform(X, rate, kernel)
                dset_real[ii] = dat.real.astype('float32')
                dset_imag[ii] = dat.imag.astype('float32')

            for dset in [dset_real, dset_imag]:
                dset.dims[0].label = 'band'
                dset.dims[1].label = 'channel'
                dset.dims[2].label = 'time'
                for val, name in ((min_freqs, 'min frequency'), (max_freqs, 'max frequency')):
                    if name not in f.keys():
                        f[name] = val
                    dset.dims.create_scale(f[name], name)
                    dset.dims[0].attach_scale(f[name])

        else:
            dset_real = f.create_dataset('X_real', (len(cts), X.shape[0], X.shape[1]),
                                         'float32', compression="gzip")
            dset_imag = f.create_dataset('X_imag', (len(cts), X.shape[0], X.shape[1]),
                                         'float32', compression="gzip")
            for ii, (ct, sd) in enumerate(tqdm(zip(cts, sds),
                                          'applying Hilbert transform',
                                          total=len(cts))):
                kernel = gaussian(X, rate, min_f, max_f)
                dat = aht.apply_hilbert_transform(X, rate, kernel)
                dset_real[ii] = dat.real.astype('float32')
                dset_imag[ii] = dat.imag.astype('float32')

            for dset in [dset_real, dset_imag]:
                dset.dims[0].label = 'filter'
                dset.dims[1].label = 'channel'
                dset.dims[2].label = 'time'
                for val, name in ((cts, 'filter_center'), (sds, 'filter_sigma')):
                    if name not in f.keys():
                        f[name] = val
                    dset.dims.create_scale(f[name], name)
                    dset.dims[0].attach_scale(f[name])

        f.attrs['sampling_rate'] = rate


if __name__=='__main__':
    main()
