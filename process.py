#%%
from os import walk
import os.path
import re

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import kurtosis
from scipy.stats import beta
import samplerate
import matplotlib as mpl
from matplotlib import gridspec

# konfiguracja
verbose = True
regen_intermediate=False
small_run = True # testy tylko na 2 plikach

test_files = ['brak_gaz_k_01_', 'brak_k_01_']

path_data = ".\data\RAW"
path_photo = ".\data\photo"
path_interm = ".\data\interm"
ext_P = "4.txt"
ext_U = "1.txt"
ext_I = "0.txt"
ext_G = "3.txt"

N_ord = 4 # rząd filtra
obcfreq = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
fs_raw = 250e+3
fs = 50e+3
resample_ratio = fs/fs_raw
ws = 2048 # szerokoś okna w samplach
ol = 0.25 # zakładka okna <0:1)

statf = [
    lambda w: kurtosis(w), # kurt
    #lambda w: beta.fit(w)[0], # beta a param
    #lambda w: beta.fit(w)[1],  # beta b param 
    # lambda w: # rice freq
]

#%%

# definicje funkcji
def _showfilter(sos, freq, fs, ax):
    wn = 8192
    w = np.zeros([wn])
    h = np.zeros([wn], dtype=np.complex_)

    w[:], h[:] = signal.sosfreqz(
        sos,
        worN=wn,
        whole=False,
        fs=fs)

    ax.semilogx(w, 20 * np.log10(abs(h) + np.finfo(float).eps), 'b')
    ax.grid(which='major')
    ax.grid(which='minor', linestyle=':')
    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    ax.set_title('Second-Order Sections - Butterworth Filter')
    plt.xlim(8, 32000)
    plt.ylim(-20, 1)
    ax.set_xticks([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(['16', '31.5', '63', '125', '250', '500',
                        '1k', '2k', '4k', '8k', '16k'])


def buffer(X = np.array([]), n = 1, p = 0):
    # function from https://stackoverflow.com/a/57491913
    #buffers data vector X into length n column vectors with overlap p
    #excess data at the end of X is discarded
    n = int(n) #length of each data vector
    p = int(p) #overlap of data vectors, 0 <= p < n-1
    L = len(X) #length of data to be buffered
    m = int(np.floor((L-n)/(n-p)) + 1) #number of sample vectors (no padding)
    data = np.zeros([n,m]) #initialize data matrix
    startIndexes = range(0,L-n,n-p)
    columns = range(0,m)
    for startIndex,column in zip(startIndexes, columns):
        data[:,column] = X[startIndex:startIndex + n] #fill in by column
    return data, np.array(startIndexes) + np.floor(n/2) 


# jedziemy...

# zbierz nazwy plików i ekstrachuj początki
if small_run:
    files = test_files
else:
    (_, _,  files) = next(walk(path_data))
    files = [re.split('0[0-4]', f)[0] for f in files]
    files = list(set(files))
#%%

for fn in files:
    if verbose:
        print(f"przetwarzany plik: {fn}")
    
    # zassij dane do dataframów
    picklename = os.path.join(path_interm, fn + ".pkl")

    if not os.path.exists(picklename) or regen_intermediate:
        df = pd.read_csv(os.path.join(path_data, fn + ext_P), names=["p"], header=None)
        df['U'] = pd.read_csv(os.path.join(path_data, fn + ext_U), names=["U"], header=None)
        df['I'] = pd.read_csv(os.path.join(path_data, fn + ext_I), names=["I"], header=None)
        if os.path.exists(os.path.join(path_data, fn + ext_G)):
            df['G'] = pd.read_csv(os.path.join(path_data, fn + ext_G), names=["G"], header=None)
        
        # wstępny downsampling do 25e+3 Sps
        data_np = df.to_numpy()
        data_np = samplerate.resample(data_np, resample_ratio, converter_type='sinc_best', verbose=verbose)
        df = pd.DataFrame(data_np, columns=df.columns)

        if verbose:
            print(f"pikluję downsamplowane sygnały jako: {picklename}")
        df.to_pickle(picklename)
    else:
        if verbose:
                print(f"ładuję intermed. z dysku: {picklename}")
        df = pd.read_pickle(picklename)
    
    # generowanie filtrów jako systemów 2. rzędu i filtracja
    X = df['p'].to_numpy()
    Xob = np.zeros((len(X), len(obcfreq)))

    for oct in range(len(obcfreq)):
        bwf = signal.butter(
            N=N_ord, 
            Wn=np.array([obcfreq[oct]/sqrt(2), obcfreq[oct]*sqrt(2)]) / (fs / 2) ,
            btype='bandpass',
            analog=False,
            output='sos' )
        # if verbose:
        #     _showfilter(bwf, obcfreq[oct], fs, ax)
        
        Xob[:, oct] = signal.sosfilt(bwf, X)

    # half-window 0 padding before stats calculation to align stats with other variables
    Xob = np.vstack([np.zeros((int(np.floor(ws/2)), np.shape(Xob)[1])), Xob])    
    # generowanie wartości cech
    m = int(np.floor((np.size(Xob, 0)-ws)/(ws*(1-ol))) + 1) #liczba okien
    #Y shape: (m windows, oct octaves, statf statistics)
    Y = np.zeros((m, len(obcfreq), len(statf)))
    
    
    for ioct in range(len(obcfreq)):
        if verbose:
            print(f"processing {obcfreq[ioct]}Hz band")
        Xbuf, centralsamps = buffer(Xob[:, ioct], ws, ws*ol)
        for isf in range(len(statf)):
            for iwin in range(m):
                Y[iwin, ioct, isf] = (statf[isf])(Xbuf[:, iwin])

    
    # for isf in range(len(statf)):
    #     plt.imshow(
    #         Y[:,:,isf].transpose(), 
    #         cmap='viridis',
    #         norm=mpl.colors.Normalize(vmin=Y[:,:,isf].min(), vmax=Y[:,:,isf].max()), 
    #         aspect='auto')
    #     plt.show()

# end of file loop

# %%
# plotting playground
plot_style = {
    'linewidth': 0.5,
    }
nrow = 5
ncol = 1

fig, axes = plt.subplots(
    nrow, ncol,
    gridspec_kw=dict(wspace=0.0, hspace=0.0,
                     top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                     left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1)),
    figsize=(6.3, 10),
    sharex='col'
    )
axes[0].imshow(
    Y[:,:,0].transpose(), 
    cmap='viridis',
    norm=mpl.colors.Normalize(vmin=Y[:,:,isf].min(), vmax=Y[:,:,isf].max()), 
    aspect='auto',
    origin='lower',
    extent = [0 , len(df)/fs, 1 , 11]
    )
axes[0].set_ylabel("oktawa")
axes[1].plot(df.index.to_numpy()/fs, df['I'], **plot_style)
axes[1].set_ylabel("I, A")
axes[2].plot(df.index.to_numpy()/fs, df['U'], **plot_style)
axes[2].set_ylabel("U, V")
lico = plt.imread(os.path.join(path_photo, fn + "lico.jpg"))
axes[3].imshow(
    lico,
    extent = [0 , len(df)/fs, -1 , 1]
    )
axes[3].set_ylabel("lico")
gran = plt.imread(os.path.join(path_photo, fn + "gran.jpg"))
axes[4].imshow(
    gran, 
    extent = [0 , len(df)/fs, -1 , 1]
    )
axes[4].set_ylabel("grań")

axes[-1].set_xlabel("t, s")
for ax in axes:
    ax.grid(axis='x', which='both')
# %%
