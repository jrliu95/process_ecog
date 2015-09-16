__author__ = 'David Conant, Jesse Livezey'

import re, tables, os, pdb, glob, csv
import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.io import loadmat
import pandas as pd

import HTK


def htk_to_hdf5(pth, blocks, task, align_window=None, dtype='HG'):
    """
    Process task data into segments with labels.

    Parameters
    ----------
    pth : str
        Path to subject folder.
    blocks : list of ints
        Blocks from which data is gathered.
    token : list of str
        Tokens to be extracted.
    align_window : list of two ints
        Time window in seconds around each token.
    dtype : str
        Type of data to use.

    Returns
    D : dict
        Dictionary containing tokens as keys and data as array.
    anat : str
        Anatomy of data channels
    start_times : dict
        Dictionary of start times per token.
    stop_times : dict
        Dictionary of stop times per token.
    -------
    """

    if task == 'CV':
        tokens = sorted(['baa', 'bee', 'boo', 'daa', 'dee', 'doo', 'faa', 'fee', 'foo',
                         'gaa', 'gee', 'goo', 'haa', 'hee', 'hoo', 'kaa', 'kee', 'koo',
                         'laa', 'lee', 'loo', 'maa', 'mee', 'moo', 'naa', 'nee', 'noo',
                         'paa', 'pee', 'poo', 'raa', 'ree', 'roo', 'saa', 'shaa', 'shee',
                         'shoo', 'see', 'soo', 'taa', 'thaa', 'thee', 'thoo', 'tee',
                         'too', 'vaa', 'vee', 'voo', 'waa', 'wee', 'woo','yaa','yee',
                         'yoo', 'zaa', 'zee', 'zoo'])
    else:
        raise ValueError("task must of one of ['CV'].")

    if dtype not in ['HG']:
        raise ValueError("dtype must be one of ['HG'].")

    if align_window is None:
        align_window = np.array([-1., 1.])

    def block_str(blocks):
        rval = 'blocks_'
        for block in blocks:
            rval += str(block) + '_'
        return rval

    def align_window_str(align_window):
        rval = 'align_window_' + str(align_window[0]) + '_to_'+ str(align_window[1])
        return rval

    subject = pth.split("/")[-1]
    fname = os.path.join(pth, 'hdf5', (subject + '_' + block_str(blocks)
                                       + task + '_' + dtype + '_'
                                       + align_window_str(align_window) + '.h5'))

    if not os.path.isfile(fname):
        D = dict((token, np.array([])) for token in tokens)
        stop_times = dict((token, np.array([])) for token in tokens)
        start_times = dict((token, np.array([])) for token in tokens)
        for iblock, block in enumerate(blocks):
            print 'Processing block ' + str(block)
            blockname = subject + '_B' + str(block)
            blockpath = os.path.join(pth, blockname)
            # Convert parseout to dataframe
            textgrid_path = os.path.join(blockpath, blockname + '_transcription_final.TextGrid')
            lab_path = os.path.join(blockpath, blockname + '_transcription_final.lab')
            if os.path.isfile(textgrid_path):
                parseout = parseTextGrid(textgrid_path)
            elif os.path.isfile(lab_path):
                parseout = parseLab(lab_path)
            else:
                raise ValueError("Transcription not found at: "
                                 + str(textgrid_path) + " or: "
                                 + str(lab_path))

            df = make_df(parseout, block, subject)

            for ind, token in enumerate(tokens):
                match = [token in t for t in df['label']]
                event_times = df['start'][match & (df['mode'] == 'speak')]
                stop = df['stop'][match & (df['mode'] == 'speak')].values
                start = df['start'][match & (df['mode'] == 'speak')].values

                stop_times[token] = (np.hstack((stop_times[token], stop.astype(float))) if 
                                     stop_times[token].size else stop.astype(float))
                start_times[token] = (np.hstack((start_times[token], start.astype(float))) if
                                      start_times[token].size else start.astype(float))
                D[token] = (np.dstack((D[token], run_makeD(blockpath, event_times,
                                                           align_window, dt=dtype))) if 
                            D[token].size else run_makeD(blockpath, event_times, align_window, dt=dtype))

        save_hdf5(fname, D, tokens)
    else:
        print 'File found; Loading...'
        D, tokens = load_hdf5(fname)

    anat = load_anatomy(pth)
    return (D)

def make_df(parseout, block, subject):
    keys = sorted(parseout.keys())
    datamat = [parseout[key] for key in keys]
    df = pd.DataFrame(np.vstack(datamat).T, columns=keys)
    df = df[df['tier'] == 'word']

    # Get rid of superfluous columns
    df = df[['label','start', 'stop']]

    # Pull mode from label and get rid of number
    df['mode'] = ['speak' if l[-1] == '2' else 'listen' for l in df['label']]
    df['label'] = df['label'].apply(lambda x: x[:-1])

    df['label'] = df['label'].astype('category')
    df['mode'] = df['mode'].astype('category')
    df['block'] = block
    df['subject'] = subject

    return df


def run_makeD(blockpath, times, t_window, dt, zscr='whole'):

    def HG():
        bad_electrodes = loadBadElectrodes(blockpath) -1
        bad_times = np.array([]) #loadBadTimes(blockpath)
        hg, fs_hg = loadHG(blockpath)

        hg = hg[:256]

        if zscr is 'whole':
            hg = stats.zscore(hg, axis=1)
        elif zscr is '30s':
            raise NotImplementedError
            for t in range(hg.shape[1]):
                trange = 5


        D = makeD(hg, fs_hg, times, t_window, bad_times=bad_times, bad_electrodes=bad_electrodes)

        return D

    def form():
        F = loadForm(blockpath)
        D = makeD(F, 100, times, t_window, bad_times=np.array([]), bad_electrodes=np.array([]))

        return D

    options = {'HG' : HG,
               'form' : form}

    D = options[dt]()

    return D

def makeD(data, fs_data, times, t_window=None, bad_times=None, bad_electrodes=None):
    """
    Aligns data to time. Assumes constant sampling frequency

    Inputs:

    Variable      Description                     Form                            Units
    =========================================================================================
    data          data to be time-aligned         np.array(n_elects x n_time)
    fs_data       frequency of data seconds       double                          seconds
    times         times to align the data to      np.array(1 x n_times)           seconds
    t_window      window around alignment time    np.array(1 x 2)                 seconds
                  (before is -ive)
    bad_times     times when there are artifacts  np.array(n_bad_times x 2)       seconds
    bad_electrodes list of bad electrodes         list of channel numbers
                   starting at 0

    Output:

    D             Data aligned to times           np.array(n_elects x n_time_win x n_times)
    """
    if t_window is None:
        t_window = np.array([-1., 1.])

    #malloc
    D = nans((data.shape[0], np.ceil(np.diff(t_window)*fs_data), len(times)))
    tt_data = np.arange(data.shape[1])/fs_data

    try:
        for itime, time in enumerate(times):
            this_data = data[:,isin(tt_data, t_window + time)]
            D[:,:this_data.shape[1],itime] = this_data

    except:
        pdb.set_trace()
    if bad_times.any():
        good_trials = [not np.any(np.logical_and(bad_times,np.any(is_overlap(t_window + time, bad_times)))) for time in times]
        D = D[:,:,good_trials]

    if len(bad_electrodes):
        bad_electrodes = bad_electrodes[bad_electrodes < D.shape[0]]
        D[bad_electrodes,:,:] = np.nan

    return D


def loadHG(blockpath):
    htk_path = os.path.join(blockpath, 'HilbAA_70to150_8band')
    HTKout = HTK.readHTKs(htk_path)
    hg = HTKout['data']
    fs_hg = HTKout['sampling_rate']/10000 # frequency in Hz

    return(hg, fs_hg)

def loadForm(blockpath):
    fname = glob.glob(os.path.join(blockpath + 'Analog', '*.ifc_out.txt'))
    F = []
    with open(fname[0]) as tsv:
        for column in zip(*[line for line in csv.reader(tsv, dialect="excel-tab")]):
            F.append(column)
    F = np.array(F)
    return F

def open_D_table(fname, tokens):
    with tables.openFile(fname) as tf:
        D = {}
        for n in tokens:
            D[n] = tf.root.__getattr__(n).read()
    return D

def parseTextGrid(fname):
    """
    Reads in a TextGrid (used by Praat) and returns a dictionary with the events
    contained within, as well as their times, labels, and hierarchy.
    Assumes a 2 tier textgrid corresponding to words and phonemes

    Parameters:
    fname: filename of the TextGrid

    Returns:
    events (a dictionary) with keys:
        label: an array of strings identifying each event according to the utterance
        start: an array of the start times for each event
        stop: an array of the stop times for each event
        tier: an array specifying the tier (phoneme or word) for each event
        contains: an array specifying the phonemes contained within each event
        contained_by: an array specifying the words contiaining each event
        position: the position of each phoneme within the word
    """

    with open(fname) as tg:
        content = tg.readlines()

    label = []
    start = []
    stop = []
    tier = []


    t = 0
    c = -1
    tiers = ['phoneme', 'word', 'phrase']
    #Loop through each line of the text file
    for line in content:
        c = c + 1
        if 'item [2]:' in line: #iterate tier as they pass
            t = 1
        if 'item [3]:' in line:
            t = 2
        if 'text =' in line:
            if line[20:-3] == 'sp' or not line[20:-3]:
                continue

            label.append(line[20:-3])
            start.append(float(content[c-2][19:-2]))
            stop.append(float(content[c-1][19:-2]))
            tier.append(tiers[t])
    label = np.array(label)
    start = np.array(start)
    stop = np.array(stop)
    tier = np.array(tier)

    phones = np.where(tier == 'phoneme')[0]
    words = np.where(tier == 'word')[0]

    contained_by = [-1]*label.size
    contains = [-1]*label.size
    position = np.ones(label.size)*-1


    #Determine hierarchy of events
    for ind in words:
        if t == 1: #If no phrase tier, word is highest tier
            position[ind] = 1
            contained_by[ind] = -1;

        #Find contained phonemes
        lower = start[ind] - 0.01
        upper = stop[ind] + 0.01
        startCandidates = np.where(start >= lower)[0]
        stopCandidates = np.where(stop <= upper)[0]
        intersect = np.intersect1d(startCandidates,stopCandidates)
        cont = list(intersect)
        cont.remove(ind)
        contains[ind] = cont
        for i in cont:
            contained_by[i] = ind


    for ind in phones:
        #Find phonemes in the same word, position is order in list
        sameWord = np.where(np.asarray(contained_by) == contained_by[ind])[0]
        position[ind] = np.where(sameWord == ind)[0] + 1


    contains = np.asarray(contains, dtype=object)
    contained_by = np.asarray(contained_by, dtype=object)

    events = {'label': label, 'start': start, 'stop': stop,
              'tier': tier, 'contains': contains,
              'contained_by': contained_by, 'position': position}
    return events


def parseLab(fname):
    """
    Reads a 'lab' transcript and returns a dictionary with the events
    contained within, as well as their times, labels, and hierarchy.

    Parameters
    ----------
    fname: filename of the 'lab' transcript

    Returns:
    events (a dictionary) with keys:
        label: an array of strings identifying each event according to the utterance
        start: an array of the start times for each event
        stop: an array of the stop times for each event
        tier: an array specifying the tier (phoneme or word) for each event
        contains: an array specifying the phonemes contained within each event
        contained_by: an array specifying the words contiaining each event
        position: the position of each phoneme within the word
    """
    start = []
    stop  = []
    tier = []
    position = []
    contains = []
    contained_by = []
    label = []
    with open(fname) as lab:
        content = lab.readlines()

    for line in content:
        token = "".join(re.findall("[a-z]",line)+[line[-2]])
        isplosive = token[0] in ['d','t','b','p','g','k','gh']
        if (isplosive and '4' in token) or (not isplosive and '3' in token):
            start.append(line[0:line.find(' ')])
            stop.append(line[line.find(' ')+1:line.find(' ',line.find(' ')+1)])
            start[-1] = float(start[-1])/1E7
            stop[-1] = float(stop[-1])/1E7
            tier.append('word')
            position.append(-1)
            contains.append(-1)
            contained_by.append(-1)
            #label.append(token)
            label.append(token[:-1] + '2') #For textgrid convention
    label = np.array(label)
    start = np.array(start)
    stop = np.array(stop)
    tier = np.array(tier)
    contains = np.asarray(contains, dtype=object)
    contained_by = np.asarray(contained_by, dtype=object)
    events = {'label': label, 'start': start, 'stop': stop, 'tier': tier,
              'contains': contains, 'contained_by': contained_by, 'position': position}
    return events

def load_anatomy(subj_dir):
    anatomy_filename = glob.glob(os.path.join(subj_dir, '*_anat.mat'))
    elect_labels_filename = glob.glob(os.path.join(subj_dir, 'elec_labels.mat'))

    if anatomy_filename:
        anatomy = sp.io.loadmat(anatomy_filename[0])
        electrode_labels = np.array([item[0][0] if len(item[0]) else '' for item in anatomy['electrodes'][0]])

    elif elect_labels_filename:
        a = sp.io.loadmat(os.path.join(subj_dir, 'elec_labels.mat'))
        electrode_labels = np.array([ elem[0] for elem in a['labels'][0]])

    else:
        electrode_labels = ''

    return electrode_labels




def save_table_file(filename, filedict):
    """Saves the variables in [filedict] in a hdf5 table file at [filename].
    """
    with tables.openFile(filename, mode="w", title="save_file") as hf:
        for vname, var in filedict.items():
            hf.createArray("/", vname, var)


def loadBadElectrodes(blockpath):
    a = []
    with open(os.path.join(blockpath, 'Artifacts', 'badChannels.txt'),'rt') as f:
        rd = csv.reader(f, delimiter=' ')
        for line in rd:
            a = line

    a = filter(None, a) # remove spaces
    a = np.array([int(x) for x in list(a)])

    return a

def nans(shape, dtype=float):
    """
    Create np.array of nans

    :param shape: tuple, dimensions of array
    :param dtype:
    :return:
    """
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def is_overlap(time_window, times_window_array):
    """
    Does time_window overlap with the time windows in times_window_array. Used for bad time segments
    :param times: np.array(1,2)
    :param times_array: np.array(x,2)
    :return: TF

    """
    def overlap(tw1,tw2):
        return not ((tw1[1] < tw2[0]) | (tw1[0] > tw2[1]))

    return [overlap(time_window,this_time_window) for this_time_window in times_window_array]

def isin(tt, tbounds):
    """
    util: Is time inside time window(s)?

    :param tt:      1 x n np.array        time counter
    :param tbounds: k, 2  np.array   time windows

    :return:        1 x n bool          logical indicating if time is in any of the windows
    """
    #check if tbounds in np.array and if not fix it
    tbounds = np.array(tbounds)

    tf = np.zeros(tt.shape, dtype = 'bool')

    if len(tbounds.shape) is 1:
        tf = (tt > tbounds[0]) & (tt < tbounds[1])
    else:
        for i in range(tbounds.shape[0]):
            tf = (tf | (tt > tbounds[i,0]) & (tt < tbounds[i,1]))
    return tf
