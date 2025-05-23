# from pycbc.noise.gaussian import frequency_noise_from_psd, noise_from_psd
from scripts.wosa import wosa
from gwpy.timeseries import TimeSeriesDict
import numpy as np
import scipy.signal

# The code is inspired from 
# https://pycbc.org/pycbc/latest/html/_modules/pycbc/noise/gaussian.html#frequency_noise_from_psd
# def psdTdiNoise(tdi_file_path, channel="X", window="hann", nper=4096, noverlap=None, average="mean", scaling='density', seed=None): 
#     obs = TimeSeriesDict.read(tdi_file_path)
#     data = obs[channel].value
#     dt  = obs[channel].dt.value
#     fs  = 1.0 / dt

#     f, psd = wosa(
#         x=data,
#         fs=fs,
#         window=window,
#         nperseg=nper,
#         noverlap=noverlap,
#         scaling=scaling,
#         average=average,
#     )   

#     delta_f = f[1] - f[0]

#     sigma = 0.5 * (psd / delta_f) ** (0.5)
#     if seed is not None:
#         numpy.random.seed(seed)
#     # sigma = sigma.numpy()
#     # dtype = complex_same_precision_as(psd)

#     not_zero = (sigma != 0)

#     sigma_red = sigma[not_zero]
#     noise_re = numpy.random.normal(0, sigma_red)
#     noise_co = numpy.random.normal(0, sigma_red)
#     noise_red = noise_re + 1j * noise_co

#     noise = numpy.zeros(len(sigma), dtype=numpy.complex128)
#     noise[not_zero] = noise_red**2
#     # noverlap  = nper // 2
#     # psd = obs.psd(nper, noverlap)
#     # psd_noise = noise_from_psd(psd)
#     # psd_freq_noise = frequency_noise_from_psd(psd)


#     return psd, f, noise

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


    ## Old: 
    # delta_f = f[1] - f[0]
    # sigma = 0.5 * (psd / delta_f) ** (0.5)
    # sigma  = 0.5 * np.sqrt(psd / delta_f)

    # ## New: 
    # # --- constants -------------------------------------------------------
    # N      = nper                       # the FFT length Welch is using
    # win    = scipy.signal.get_window(window, N)
    # U      = (win**2).mean()            # coherent power gain of the window
    # delta_f = fs / N                    # safer than f[1]-f[0]

    # # --- per-bin sigma -------------------------------------------------------
    # sigma = np.sqrt(0.5 * psd * delta_f / U)   # include window correction
    # sigma[0]      *= np.sqrt(2)                # DC bin (no factor-2 later)
    # sigma[-1]     *= np.sqrt(2) if N % 2 == 0 else 1   # Nyquist if present


    ## Newer: 
    # --- constants -------------------------------------------------------
    N  = nper                    # FFT length used by Welch
    fs = 1.0 / dt                # sampling rate

    # --- per-bin standard deviation -------------------------------------
    sigma = np.sqrt(psd * fs * N / 4)   # k = 1 … N/2-1
    sigma[0]  = np.sqrt(psd[0]  * fs * N / 2)      # DC (real only)
    if N % 2 == 0:                                 # Nyquist if present
        sigma[-1] = np.sqrt(psd[-1] * fs * N / 2)


    if seed is not None:
        np.random.seed(seed)
    # sigma = sigma.numpy()
    # dtype = complex_same_precision_as(psd)

    not_zero = (sigma != 0)

    sigma_red = sigma[not_zero]
    noise_re = np.random.normal(0, sigma_red)
    noise_co = np.random.normal(0, sigma_red)
    noise_red = noise_re + 1j * noise_co

    noise = np.zeros(len(sigma), dtype=np.complex128)
    noise[not_zero] = noise_red

    # noverlap  = nper // 2
    # psd = obs.psd(nper, noverlap)
    # psd_noise = noise_from_psd(psd)
    # psd_freq_noise = frequency_noise_from_psd(psd)

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