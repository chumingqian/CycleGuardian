"""
Created on Sat May 27 15:37:50 2017
Python version of:
D. P. W. Ellis (2009). "Gammatone-like spectrograms", web resource. http://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/
On the corresponding webpage, Dan notes that he would be grateful if you cited him if you use his work (as above).
This python code does not contain all features present in MATLAB code.
Sat May 27 15:37:50 2017 Maddie Cusimano, mcusi@mit.edu 27 May 2017
"""

from __future__ import division 
import numpy as np
import scipy.signal as sps
import scipy.io.wavfile as wf
import torch
import torchaudio.transforms

# from gtg import gammatonegram
import numpy as np
from scipy.stats import norm as ssn
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt



def fft2gammatonemx(nfft, sr=20000, nfilts=64, width=1.0, minfreq=100,
                    maxfreq=10000, maxlen=1024):    
    """
    # Ellis' description in MATLAB:
    # [wts,cfreqa] = fft2gammatonemx(nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
    #      Generate a matrix of weights to combine FFT bins into
    #      Gammatone bins.  nfft defines the source FFT size at
    #      sampling rate sr.  Optional nfilts specifies the number of
    #      output bands required (default 64), and width is the
    #      constant width of each band in Bark (default 1).
    #      minfreq, maxfreq specify range covered in Hz (100, sr/2).
    #      While wts has nfft columns, the second half are all zero. 
    #      Hence, aud spectrum is
    #      fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft));
    #      maxlen truncates the rows to this many bins.
    #      cfreqs returns the actual center frequencies of each
    #      gammatone band in Hz.
    #
    # 2009/02/22 02:29:25 Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    # Sat May 27 15:37:50 2017 Maddie Cusimano, mcusi@mit.edu 27 May 2017: convert to python
    """
    
    wts = np.zeros([nfilts,nfft])
    
    #after Slaney's MakeERBFilters
    EarQ = 9.26449; minBW = 24.7; order = 1;
    
    nFr = np.array(range(nfilts)) + 1
    em = EarQ*minBW
    cfreqs = (maxfreq+em)*np.exp(nFr*(-np.log(maxfreq + em)+np.log(minfreq + em))/nfilts)-em
    cfreqs = cfreqs[::-1]
    
    GTord = 4
    ucircArray = np.array(range(int(nfft/2 + 1)))
    ucirc = np.exp(1j*2*np.pi*ucircArray/nfft);
    #justpoles = 0 :taking out the 'if' corresponding to this. 

    ERB = width*np.power(np.power(cfreqs/EarQ,order) + np.power(minBW,order),1/order);
    B = 1.019 * 2 * np.pi * ERB;
    r = np.exp(-B/sr)
    theta = 2*np.pi*cfreqs/sr
    pole = r*np.exp(1j*theta)
    T = 1/sr
    ebt = np.exp(B*T); cpt = 2*cfreqs*np.pi*T;  
    ccpt = 2*T*np.cos(cpt); scpt = 2*T*np.sin(cpt);
    A11 = -np.divide(np.divide(ccpt,ebt) + np.divide(np.sqrt(3+2**1.5)*scpt,ebt),2); 
    A12 = -np.divide(np.divide(ccpt,ebt) - np.divide(np.sqrt(3+2**1.5)*scpt,ebt),2);
    A13 = -np.divide(np.divide(ccpt,ebt) + np.divide(np.sqrt(3-2**1.5)*scpt,ebt),2); 
    A14 = -np.divide(np.divide(ccpt,ebt) - np.divide(np.sqrt(3-2**1.5)*scpt,ebt),2);
    zros = -np.array([A11, A12, A13, A14])/T;
    wIdx = range(int(nfft/2 + 1))  
    gain = np.abs((-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) - np.sqrt(3 - 2**(3/2))*  np.sin(2*cfreqs*np.pi*T))) *(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) + np.sqrt(3 - 2**(3/2)) *  np.sin(2*cfreqs*np.pi*T)))*(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) -  np.sqrt(3 + 2**(3/2))*np.sin(2*cfreqs*np.pi*T))) *(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) + np.sqrt(3 + 2**(3/2))*np.sin(2*cfreqs*np.pi*T))) /(-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cfreqs*np.pi*T) +  2*(1 + np.exp(4*1j*cfreqs*np.pi*T))/np.exp(B*T))**4);
    #in MATLAB, there used to be 64 where here it says nfilts:
    wts[:, wIdx] =  ((T**4)/np.reshape(gain,(nfilts,1))) * np.abs(ucirc-np.reshape(zros[0],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[1],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[2],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[3],(nfilts,1)))*(np.abs(np.power(np.multiply(np.reshape(pole,(nfilts,1))-ucirc,np.conj(np.reshape(pole,(nfilts,1)))-ucirc),-GTord)));           
    wts = wts[:,range(maxlen)];   
    
    return wts, cfreqs

def gammatonegram(x,sr=20000,twin=0.025,thop=0.010,N=64,
                  fmin=50,fmax=10000,width=1.0):
    """
    # Ellis' description in MATLAB: 
    # [Y,F] = gammatonegram(X,SR,N,TWIN,THOP,FMIN,FMAX,USEFFT,WIDTH)
    # Calculate a spectrogram-like time frequency magnitude array
    # based on Gammatone subband filters.  Waveform X (at sample
    # rate SR) is passed through an N (default 64) channel gammatone 
    # auditory model filterbank, with lowest frequency FMIN (50) 
    # and highest frequency FMAX (SR/2).  The outputs of each band 
    # then have their energy integrated over windows of TWIN secs 
    # (0.025), advancing by THOP secs (0.010) for successive
    # columns.  These magnitudes are returned as an N-row
    # nonnegative real matrix, Y.
    # WIDTH (default 1.0) is how to scale bandwidth of filters 
    # relative to ERB default (for fast method only).
    # F returns the center frequencies in Hz of each row of Y
    # (uniformly spaced on a Bark scale).
      
    # 2009/02/23 DAn Ellis dpwe@ee.columbia.edu
    # Sat May 27 15:37:50 2017 Maddie Cusimano mcusi@mit.edu, converted to python
    """

    #Entirely skipping Malcolm's function, because would require
    #altering ERBFilterBank code as well. 
    #i.e., in Ellis' code: usefft = 1
    assert(x.dtype == 'int16')

    # How long a window to use relative to the integration window requested
    winext = 1;
    twinmod = winext*twin;
    nfft = int(2**(np.ceil(np.log(2*twinmod*sr)/np.log(2))))
    nhop = int(np.round(thop*sr))
    nwin = int(np.round(twinmod*sr))
    [gtm,f] = fft2gammatonemx(nfft, sr, N, width, fmin, fmax, int(nfft/2+1))
    # perform FFT and weighting in amplitude domain
    # note: in MATLAB, abs(spectrogram(X, hanning(nwin), nwin-nhop, nfft, SR))
    #                  = abs(specgram(X,nfft,SR,nwin,nwin-nhop))
    # in python approx = sps.spectrogram(x, fs=sr, window='hann', nperseg=nwin, 
    #                    noverlap=nwin-nhop, nfft=nfft, detrend=False, 
    #                    scaling='density', mode='magnitude')
    plotF, plotT, Sxx = sps.spectrogram(x, fs=sr, window='hann', nperseg=nwin, 
                                noverlap=nwin-nhop, nfft=nfft, detrend=False, 
                                scaling='density', mode='magnitude')
    y = (1/nfft)*np.dot(gtm,Sxx)
    
    return y, f




def gammatonegram_torch(x, sr=20000, nfft= None, nhop=None, nwin=None, N=64, fmin=50, fmax=10000, width=1.0):

    # Entirely skipping Malcolm's function, because would require
    # altering ERBFilterBank code as well.
    # i.e., in Ellis' code: usefft = 1
    # assert (x.dtype == 'int16')

    # How long a window to use relative to the integration window requested
    # winext = 1;
    #twinmod = winext * twin;
    # nfft = int(2 ** (np.ceil(np.log(2 * twinmod * sr) / np.log(2))))
    # nhop = int(np.round(thop * sr))
    # nwin = int(np.round(twinmod * sr))

    [gtm, f] = fft2gammatonemx(nfft, sr, N, width, fmin, fmax, int(nfft / 2 + 1))
    spec_torch_fun = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=nwin, hop_length= nhop, normalized= True)
    spec_torch = spec_torch_fun(torch.Tensor(x))

    spec_torch = spec_torch.numpy()
    y = (1 / nfft) * np.dot(gtm, spec_torch)

    return y, f


def wheeze_gammatonegram_torch(x, sr=20000, nfft= None, nhop=None, nwin=None, N=64, fmin=50, fmax=10000, width=1.0):
    print(" gtg.py:   wheeze gamma torch  fun\n ")

    [gtm, f] = fft2gammatonemx(nfft, sr, N, width, fmin, fmax, int(nfft / 2 + 1))
    print(" gtg.py:  gtm  filters bank  shape should be  [64,  257] \n", gtm.shape)
    spec_torch_fun = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=nwin, hop_length= nhop, normalized= True)
    spec_torch = spec_torch_fun(torch.Tensor(x))
    print(" gtg.py:  torch spectrogram  shape should be  [257, 640] \n", spec_torch.shape)

    wheeze_mask = torch.zeros((257, 640))
    wheeze_mask[7:27, :] = 1  # mask  for the Rhonchus, 50Hz-200Hz,
    wheeze_mask[49:72, :] = 1  # mask for the wheeze , 400Hz, range 380~550 Hz;

    wheeze_mask_spec = torch.mul(spec_torch, wheeze_mask)
    print(" gtg.py:   wheeze mask mel  spectrogram  shape should be  [257, 640] \n", wheeze_mask_spec.shape)

    wheeze_mask_spec = wheeze_mask_spec.numpy()
    y = (1 / nfft) * np.dot(gtm, wheeze_mask_spec)
    print(" gtg.py:   wheeze mask mel  spectrogram  shape should be  [64, 640] \n", y.shape)
    return y, f


def crackle_gammatonegram_torch(x, sr=20000, nfft= None, nhop=None, nwin=None, N=64, fmin=50, fmax=10000, width=1.0):
    print(" gtg.py:   wheeze gamma torch  fun\n ")

    [gtm, f] = fft2gammatonemx(nfft, sr, N, width, fmin, fmax, int(nfft / 2 + 1))
    print(" gtg.py:  gtm  filters bank  shape should be  [64,  257] \n", gtm.shape)
    spec_torch_fun = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=nwin, hop_length= nhop, normalized= True)
    spec_torch = spec_torch_fun(torch.Tensor(x))
    print(" gtg.py:  torch spectrogram  shape should be  [257, 640] \n", spec_torch.shape)

    wheeze_mask = torch.zeros((257, 640))
    wheeze_mask[7:27, :] = 1  # mask  for the Rhonchus, 50Hz-200Hz,
    wheeze_mask[49:72, :] = 1  # mask for the wheeze , 400Hz, range 380~550 Hz;

    wheeze_mask_spec = torch.mul(spec_torch, wheeze_mask)
    print(" gtg.py:   wheeze mask mel  spectrogram  shape should be  [257, 640] \n", wheeze_mask_spec.shape)

    wheeze_mask_spec = wheeze_mask_spec.numpy()
    y = (1 / nfft) * np.dot(gtm, wheeze_mask_spec)
    print(" gtg.py:   wheeze mask mel  spectrogram  shape should be  [64, 640] \n", y.shape)
    return y, f





#sxx, center_frequencies = gammatonegram_from_stft_spec(sfft_spec, sr=sampling_rate, nfft=n_fft, nwin=n_win, nhop=n_hop, fmin=0,fmax=3500)


def gammatonegram_from_stft_spec(cur_spectrogram, sr, nfft= None, nwin= None, nhop= None, N =112, fmin=0,fmax=3500, width= 1.0):
    # Entirely skipping Malcolm's function, because would require
    # altering ERBFilterBank code as well.
    # i.e., in Ellis' code: usefft = 1

    [gtm, f] = fft2gammatonemx(nfft, sr, N, width, fmin, fmax, int(nfft / 2 + 1))
    # spec_torch_fun = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=nwin, hop_length= nhop, normalized= True)
    spec_torch = cur_spectrogram


    spec_torch = spec_torch.cpu().numpy()
    y = (1 / nfft) * np.dot(gtm, spec_torch)

    return y, f


"""
Example of using python gammatonegram code 
mcusi@mit.edu, 2018 Sept 24
"""


def gtg_in_dB(sound, sampling_rate, log_constant=1e-80, dB_threshold=-50.0):
    """ Convert sound into gammatonegram, with amplitude in decibels"""
    # sxx, center_frequencies = gammatonegram(sound, sr=sampling_rate, fmin=20, fmax=int(sampling_rate / 2.))
    sxx, center_frequencies = gammatonegram_torch(sound, sr=sampling_rate, fmin=0, fmax= 4500)

    sxx[sxx == 0] = log_constant
    sxx = 20. * np.log10(sxx)  # convert to dB
    sxx[sxx < dB_threshold] = dB_threshold
    return sxx, center_frequencies

import librosa

def gtg_in_dB_torch(sound, sampling_rate, n_fft, n_win, n_hop, log_constant=1e-80, dB_threshold=-50.0):
    """ Convert sound into gammatonegram, with amplitude in decibels"""
    # sxx, center_frequencies = gammatonegram(sound, sr=sampling_rate, fmin=20, fmax=int(sampling_rate / 2.))
    sxx, center_frequencies = gammatonegram_torch(sound, sr=sampling_rate, nfft= n_fft, nwin= n_win, nhop= n_hop, fmin=20, fmax= 2000)
    sxx[sxx == 0] = log_constant
    # sxx = 20. * np.log10(sxx)  # convert to dB
    sxx = librosa.power_to_db(sxx, ref=np.max)
    sxx[sxx < dB_threshold] = dB_threshold
    return sxx, center_frequencies



import  cv2
import  cmapy

def gen_gamma_3channel(sound, sampling_rate, n_fft, n_win, n_hop, log_constant=1e-80, dB_threshold=-50.0):
    """ Convert sound into gammatonegram, with amplitude in decibels"""
    # sxx, center_frequencies = gammatonegram(sound, sr=sampling_rate, fmin=20, fmax=int(sampling_rate / 2.))
    sxx, center_frequencies = gammatonegram_torch(sound, sr=sampling_rate, nfft= n_fft, nwin= n_win, nhop= n_hop, fmin=150, fmax= 4000)
    S = sxx
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.min()) / (S.max() - S.min())
    S *= 255
    img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('magma'))
    img = cv2.flip(img, 0)

    return img, center_frequencies






def gen_Gamma_3channel(gamma_spectrogram, center_f, log_constant=1e-80, dB_threshold=-50, resz=0):
    '''
    from the  signle  channel  generate  the 3 channel;
    Parameters
    ----------
    gamma_spectrogram
    resz

    Returns
    -------
    '''

    """ Convert sound into gammatonegram, with amplitude in decibels"""
    # sxx, center_frequencies = gammatonegram(sound, sr=sampling_rate, fmin=20, fmax=int(sampling_rate / 2.))
    # gammatonegram_torch(sound, sr=sampling_rate, nfft= n_fft, nwin= n_win, nhop= n_hop, fmin=20, fmax= 2000)
    sxx, center_frequencies = gamma_spectrogram,  center_f
    sxx[sxx == 0] = log_constant
    # sxx = 20. * np.log10(sxx)  # convert to dB
    sxx = librosa.power_to_db(sxx, ref=np.max)
    sxx[sxx < dB_threshold] = dB_threshold

    S =  sxx
    S = 255 * (S - np.min(S)) / (np.max(S) - np.min(S))
    # print(" gamma spectrogram before the color map： \n", S.shape)
    img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('magma'))
    # print(" gamma spectrogram after the color map： \n", img.shape)

    height, width, _ = img.shape
    if resz > 0:
        img = cv2.resize(img, (width * resz, height * resz), interpolation=cv2.INTER_LINEAR)

    if resz == 0:
        img = cv2.flip(img, 0)
        # print(" after the flip operate,  \n", img.shape)
    # img = cv2.flip(img, 0)
    # print(" mel spectrogram after  add color channel： \n", img.shape)
    return img









def gtg_in_dB_torch_from_stft( sfft_spec, sampling_rate, n_fft, n_win, n_hop, log_constant=1e-80, dB_threshold=-50.0):
    """ Convert sound into gammatonegram, with amplitude in decibels"""
    # sxx, center_frequencies = gammatonegram(sound, sr=sampling_rate, fmin=20, fmax=int(sampling_rate / 2.))

    sxx, center_frequencies = gammatonegram_from_stft_spec(sfft_spec, sr=sampling_rate, nfft=n_fft, nwin=n_win, nhop=n_hop, fmin=0,fmax=3500)
    # sxx, center_frequencies = gammatonegram_torch(sound, sr=sampling_rate, nfft= n_fft, nwin= n_win, nhop= n_hop, fmin=0, fmax= 3500)

    sxx[sxx == 0] = log_constant
    sxx = 20. * np.log10(sxx)  # convert to dB
    sxx[sxx < dB_threshold] = dB_threshold
    return sxx, center_frequencies




"""
def loglikelihood(sxx_observation, sxx_hypothesis):
    likelihood_weighting = 5.0 #free parameter!
    loglikelihood = likelihood_weighting*np.sum(ssn.logpdf(sxx_observation, 
                                                           loc=sxx_hypothesis, scale=1))
    return loglikelihood 
"""


def gtgplot(sxx, center_frequencies, sample_duration, sampling_rate,
            dB_threshold=-50.0, dB_max=10.0, t_space=50, f_space=10):
    """Plot gammatonegram"""
    fig, ax = plt.subplots(1, 1)

    time_per_pixel = sample_duration / (1. * sampling_rate * sxx.shape[1])
    t = time_per_pixel * np.arange(sxx.shape[1])

    plt.pcolormesh(sxx, vmin=dB_threshold, vmax=dB_max, cmap='Blues')
    ax.set_ylabel('Frequency (Hz)', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.xaxis.set_ticks(range(sxx.shape[1])[::t_space])
    ax.xaxis.set_ticklabels((t[::t_space] * 100.).astype(int) / 100., fontsize=16)
    ax.set_xbound(0, sxx.shape[1])
    ax.yaxis.set_ticks(range(len(center_frequencies))[::f_space])
    ax.yaxis.set_ticklabels(center_frequencies.astype(int)[::f_space], fontsize=16)
    ax.set_ybound(0, len(center_frequencies))
    cbar = plt.colorbar(pad=0.01)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel('Amplitude (dB)', rotation=270, labelpad=15, fontsize=16)
    plt.show()
    plt.close()


'''
if __name__ == '__main__':
    sampling_rate, sound = wf.read('sample.wav')
    sxx, center_frequencies = gtg_in_dB(sound, sampling_rate)
    gtgplot(sxx, center_frequencies, len(sound), sampling_rate)
    
    
'''




