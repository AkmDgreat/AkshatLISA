# from pycbc.noise.gaussian import frequency_noise_from_psd, noise_from_psd
from scripts.wosa import wosa
from gwpy.timeseries import TimeSeriesDict
import numpy as np
import scipy.signal

# The code is modified version of code here: 
# https://pycbc.org/pycbc/latest/html/_modules/pycbc/noise/gaussian.html#frequency_noise_from_psd
# This function returns the original PSD, and the noise PSDs derived from the original PSD 
def psdTdiNoise(tdi_file_path, channel="X", window="hann", nper=4096, noverlap=None, average="mean", scaling='density', seed=None): 

    # 1) Find the PSD of the TDI time series: 
    obs = TimeSeriesDict.read(tdi_file_path)
    data = obs[channel].value
    dt  = obs[channel].dt.value
    fs  = 1.0 / dt

    f, psd = wosa(
        x=data,
        fs=fs,
        window=window,
        nperseg=nper,
        noverlap=noverlap,
        scaling=scaling,
        average=average,
    ) 
    print(f"The length of psd: {psd.size}")

    # 2) Get frequency-domain noise from this PSD

    # Old Sigma (taken from Pycbc, incorrect scaling coefficient): 
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

    # 3) inverse-fourier transform the freuqnecy-domain-noise to time-domain-noise 
    M = len(noise)           # length of FrequencySeries
    N = 2 * (M - 1)          # use this for irfft
    time_noise = np.fft.irfft(noise, n=N)
    print(f"time_domain_noise length: {len(time_noise)}")

    # 4) Find the PSD of the time-domain-noise 
    _, psd_noise = wosa(
        x=time_noise,
        fs=fs,
        window=window,
        nperseg=nper,
        noverlap=noverlap,
        scaling=scaling,
        average=average,
    )  

    print(f"Noise psd length: {len(psd_noise)}") 

    return psd, f, psd_noise