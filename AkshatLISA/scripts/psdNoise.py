# from pycbc.noise.gaussian import frequency_noise_from_psd, noise_from_psd
from scripts.wosa import wosa
from gwpy.timeseries import TimeSeriesDict
import numpy as np
import scipy.signal

# The code is modified version of code here: 
# https://pycbc.org/pycbc/latest/html/_modules/pycbc/noise/gaussian.html#frequency_noise_from_psd
def psdTdiNoise(tdi_file_path, channel="X", window="hann", nper=4096, noverlap=None, average="mean", scaling='density', seed=None): 
    obs = TimeSeriesDict.read(tdi_file_path)
    data = obs[channel].value
    dt  = obs[channel].dt.value
    fs  = 1.0 / dt

    f, psd = scipy.signal.welch(
        x=data,
        fs=fs,
        window=window,
        nperseg=nper,
        noverlap=noverlap,
        scaling=scaling,
        average=average,
    ) 

    # Old Sigma: 
    # sigma = 0.5 * np.sqrt(psd / psd.delta_f) 

    # --- per-bin standard deviation -------------------------------------
    sigma = np.sqrt(psd * fs * nper / 4)   # k = 1 … N/2-1
    sigma[0]  = np.sqrt(psd[0]  * fs * nper / 2)      # DC (real only)
    if nper % 2 == 0:                                 # Nyquist if present
        sigma[-1] = np.sqrt(psd[-1] * fs * nper / 2)

    if seed is not None:
        np.random.seed(seed)

    not_zero = (sigma != 0)

    sigma_red = sigma[not_zero]
    noise_re = np.random.normal(0, sigma_red)
    noise_co = np.random.normal(0, sigma_red)
    noise_red = noise_re + 1j * noise_co

    noise = np.zeros(len(sigma), dtype=np.complex128)
    noise[not_zero] = noise_red

    M = len(noise)           # length of FrequencySeries
    N = 2 * (M - 1)          # use this for irfft
    time_noise = np.fft.irfft(noise, n=N)

    _, psd_noise = scipy.signal.welch(
        x=time_noise,
        fs=fs,
        window=window,
        nperseg=nper,
        noverlap=noverlap,
        scaling=scaling,
        average=average,
    )   

    return psd, f, psd_noise