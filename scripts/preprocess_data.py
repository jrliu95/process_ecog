from __future__ import print_function, division

import argparse, h5py, time, os
import numpy as np
from scipy.io import loadmat

from ecog.signal_processing import resample
from ecog.signal_processing import subtract_CAR
from ecog.signal_processing import linenoise_notch
from ecog.signal_processing import hilbert_transform
from ecog.signal_processing import gaussian
from ecog.utils import load_bad_electrodes, bands

import nwbext_ecog
from pynwb import NWBHDF5IO
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from pynwb.core import DynamicTable
from pynwb.misc import DecompositionSeries


def main():
    parser = argparse.ArgumentParser(description='Preprocessing ecog data.')
    parser.add_argument('path', type=str, help="Path to the data")
    parser.add_argument('subject', type=str, help="Subject code")
    parser.add_argument('blocks', type=int, nargs='+',
                        help="Block number eg: '1'")
    parser.add_argument('-e', '--phase', default=False, action='store_true',
                        help="Apply random phases to the hilbert transform.")
    args = parser.parse_args()
    print(args)

    for block in args.blocks:
        block_path = os.path.join(args.path, args.subject,
                                  '{}_B{}.nwb'.format(args.subject, block))
        transform(block_path, phase=args.phase)


def transform(block_path, suffix=None, phase=False, total_channels=256,
              seed=20180928):
    """
    Takes raw LFP data and does the standard hilb algorithm:
    1) CAR
    2) notch filters
    3) Hilbert transform on different bands
    ...

    Saves to os.path.join(block_path, subject + '_B' + block + '_AA.h5')

    Parameters
    ----------
    block_path
    rate
    cfs: filter center frequencies. If None, use Chang lab defaults
    sds: filer standard deviations. If None, use Chang lab defaults

    takes about 20 minutes to run on 1 10-min block
    """

    rng = None
    if phase:
        rng = np.random.RandomState(seed)
    rate = 400.

    cfs = bands.chang_lab['cfs']
    sds = bands.chang_lab['sds']

    subj_path, block_name = os.path.split(block_path)
    block_name = os.path.splitext(block_path)[0]

    start = time.time()

    with NWBHDF5IO(block_path, 'r') as io:
        nwb = io.read()
        # 1e6 scaling helps with numerical accuracy
        X = nwb.acquisition['ECoG'].data[:].T * 1e6
        fs = nwb.acquisition['ECoG'].rate
        bad_elects = load_bad_electrodes(nwb)
        session_start_time = nwb.session_start_time
        ecog_subject = nwb.subject
        electrode_table = nwb.electrodes
        electrode_groups = nwb.electrode_groups
        devices = nwb.devices

    print('Load time for {}: {} seconds'.format(block_name,
                                                time.time() - start))
    print('rates {}: {} {}'.format(block_name, rate, fs))
    if not np.allclose(rate, fs):
        assert rate < fs
        X = resample(X, rate, fs)

    if X.shape[0] != total_channels:
        raise ValueError(block_name, X.shape, total_channels)

    if bad_elects.sum() > 0:
        X[bad_elects] = np.nan

    # Subtract CAR
    start = time.time()
    X = subtract_CAR(X)
    print('CAR subtract time for {}: {} seconds'.format(block_name,
                                                        time.time() - start))

    # Apply Notch filters
    start = time.time()
    X = linenoise_notch(X, rate)
    print('Notch filter time for {}: {} seconds'.format(block_name,
                                                        time.time() - start))

    # Apply Hilbert transform and store
    if suffix is None:
        suffix_str = ''
    else:
        suffix_str = '_{}'.format(suffix)
    if phase:
        suffix_str = suffix_str + '_random_phase'

    # Define NWB file path
    fname = '{}_HilbAA{}.nwb'.format(block_name, suffix_str)

    AA_path = os.path.join(block_path, fname)
    X = X.astype('float32')

    # create list of arrays (channel x time) that is the length of the frequency bands
    dset = []

    theta = None
    if phase:
        theta = rng.rand(*X.shape) * 2. * np.pi
        theta = np.sin(theta) + 1j * np.cos(theta)

    X_fft_h = None
    for ii, (cf, sd) in enumerate(zip(cfs, sds)):
        kernel = gaussian(X, rate, cf, sd)
        Xp, X_fft_h = hilbert_transform(X, rate, kernel, phase=theta, X_fft_h=X_fft_h)
        dset.append(abs(Xp).astype('float32'))

    # stack data to be (bands, channels, times)
    dset = np.stack(dset, axis=0)

    # change dimensions to work with NWB DecompositionSeries (times, channels, bands)
    dset = np.transpose(dset, (2, 1, 0))

    # create DynamicTable for labeling the frequency bands
    bands_table = DynamicTable('Hilbert_AA_bands',
                               'frequency band mean and stdev for hilbert transform',
                               id=np.arange(len(cfs)))
    bands_table.add_column('band_mean',
                           'frequency band centers',
                           data=cfs)
    bands_table.add_column('filter_stdev',
                           'frequency band stdev',
                           data=sds)

    # create the DecompositionSeries to store the spectral decomposition
    ds = DecompositionSeries('Hilbert_AA',
                             dset,
                             description='hilbert analytic amplitude',
                             metric='analytic amplitude',
                             bands=bands_table,
                             starting_time=0.0,
                             rate=rate)

    # Create new NWB file to store spectral decomposition
    decomp_nwb_file = NWBFile('Hilbert transformed ECoG data',
                              block_name.split('/')[-1],
                              session_start_time,
                              institution='University of California, San Francisco',
                              lab='Chang Lab')
                              # electrodes=electrode_table,
                              # electrode_groups=electrode_groups)
                              #devices=devices,
                              #subject=ecog_subject)

    # Add the decomposition information
    decomp_nwb_file.add_acquisition(ds)

    # Write the finished NWB file
    with NWBHDF5IO(AA_path, 'w') as f:
        f.write(decomp_nwb_file)

    print('{} finished'.format(block_name))
    print('saved: {}'.format(AA_path))


if __name__ == '__main__':
    main()
