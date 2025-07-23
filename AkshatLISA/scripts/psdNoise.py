# from pycbc.noise.gaussian import frequency_noise_from_psd, noise_from_psd
from psd_estimation_methods.wosa import wosa
import numpy as np
import scipy.signal
import random 

# The code is modified version of code here: 
# https://pycbc.org/pycbc/latest/html/_modules/pycbc/noise/gaussian.html#frequency_noise_from_psd
def time_noise_from_psd(
    psd, 
    fs, 
    nperseg, 
    seed=None
): 
    """
    Generate a time-domain noise realization whose PSD matches the input PSD.

    Parameters
    ----------
    psd : array_like
        Desired PSD.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Number of samples in each time segment (length of the segment).
    seed : int, optional
        Seed for the random number generator. If provided, the output is reproducible.

    Returns
    -------
    numpy.ndarray
        Real-valued time series of length N = 2*(len(psd)-1) 
        The resulting noise realization has a PSD that matches the input `psd`.

    Notes
    -----
    Adapted from PyCBC's `frequency_noise_from_psd`:
    https://pycbc.org/pycbc/latest/html/_modules/pycbc/noise/gaussian.html#frequency_noise_from_psd
    """

    # Old Sigma (taken from pycbc, incorrect scaling coefficient): 
    # sigma = 0.5 * np.sqrt(psd / psd.delta_f) 

    sigma = np.sqrt(psd * fs * nperseg / 4)   
    sigma[0]  = np.sqrt(psd[0]  * fs * nperseg / 2)      # DC (real only)
    if nperseg % 2 == 0:                                 # Nyquist if present
        sigma[-1] = np.sqrt(psd[-1] * fs * nperseg / 2)

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
    
    return time_noise

def n_time_noise_from_psd(
    psd, 
    fs, 
    nperseg, 
    n=100,
    seed_offset=0
):
    """
    Generates n time-domain noise realizations whose PSD matches the input PSD.

    Parameters
    ----------
    psd : array_like
        Desired PSD.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Number of samples per time segment.
    n : int, optional
        Number of noise realizations to generate.
    seed_offset : int, optional
        Integer offset used to seed the random number generator for reproducibility.

    Returns
    -------
        numpy.ndarray, shape (n, nperseg)
            A 2D array where each row is a real-valued time-series noise realization whose
            PSD matches the input `psd`.
    """
    time_noises =  np.zeros((n, nperseg))
    for i in range(n):
        time_noises[i] = time_noise_from_psd(
            psd, 
            fs, 
            nperseg, 
            seed=seed_offset + i
        )

    return time_noises

def n_noise_psds(time_noises, fs, nperseg, noverlap=None, window='hann', method='mean'):
    """
    Computes PSD of n time-domain noise realisations

    Parameters
    ----------
    time_noises : array_like, shape (n, nperseg)
        Array of n time-series noise realizations, each of length `nperseg` samples.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Number of samples per segment for WOSA
    noverlap : int
        Number of samples to overlap between adjacent segments.
    
    Returns
    -------
    f_noise : ndarray, shape (nperseg//2 + 1,)
        The frequency bins corresponding to the PSD values.
    noise_psds : ndarray, shape (n, nperseg//2 + 1)
        PSD estimates for each noise realization.  Row `i` is the PSD of `time_noises[i]`.
    """
    n = time_noises.shape[0]
    noise_psds = np.zeros((n, nperseg // 2 + 1))

    for i in range(n):
        f_noise, noisePsd, _, nseg = wosa(x=time_noises[i], fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, method=method)
        noise_psds[i] = noisePsd

    return f_noise, noise_psds, nseg

def compute_psd_noise_distribution(
    orig_psd,
    orig_f,
    noise_psds,
    f_noise,
    fraction=0.5,
):
    """
    Sample the PSD distribution of noise realizations at a given frequency and 
    compare it to the original PSD.

    Parameters
    ----------
    orig_psd : array_like, shape (M,)
        Original PSD
    orig_f : array_like, shape (M,)
        Frequencies corresponding to `orig_psd`.
    
    noise_psds : array_like, shape (n_realizations, K)
        Noise PSDs
    f_noise : array_like, shape (K,)
        Frequency bins corresponding to the columns of `noise_psds`.
    
    fraction : float, optional
        Fraction ∈ [0, 1] selecting the noise-PSD bin:
        0 → lowest nonzero frequency, 1 → highest frequency. 

    Returns
    -------
    psd_noise_vals : ndarray, shape (n,)
        PSD values of each noise realization at the selected noise frequency.
    orig_psd_val : float
        Original PSD value at the frequency in `orig_f` closest to the selected noise frequency.
    chosen_f_noise : float
        The selected noise frequency (Hz) 
    """

    # 1) pick bin index based on fraction
    if not (0 <= fraction <= 1):
        raise ValueError("`fraction` must be between 0 and 1")
    idx = int(fraction * (len(f_noise) - 1))
    chosen_f_noise = f_noise[idx]

    # 2) find the original PSD at the closest frequency in orig_f
    closest_orig_idx = np.argmin(np.abs(orig_f - chosen_f_noise))
    chosen_f_orig = orig_f[closest_orig_idx]
    orig_psd_val = orig_psd[closest_orig_idx]

    # 3) collect noise PSD values at that bin
    psd_noise_vals = noise_psds[:, idx]

    return psd_noise_vals, orig_psd_val, chosen_f_noise, chosen_f_orig

def normalized_psd_residual(orig_psd:  np.ndarray,
                            noise_psds: np.ndarray,
                            N: int) -> np.ndarray:
    """
    Down-sample `orig_psd` so it lives on the same grid as `noise_psds`
    (stride k = (len(orig_psd)-1)/(len(f_noise)-1)), then return
      (orig_on_noise – mean_noise) / (std_noise/√N).

    Parameters
    ----------
    orig_psd   : (M_fine,)   fine-grid PSD, e.g. length 10 001
    noise_psds : (n_real, M_coarse) ensemble on coarse grid, e.g. length 2 501
    N          : number of statistically-independent averages per PSD
                 (σ_SE = σ / √N)

    Returns
    -------
    residual   : (M_coarse,) normalised residual on the coarse grid
    """
    
    # --- 1. statistics from the noise ensemble -----------------------
    mean_noise = noise_psds.mean(axis=0)            # shape (M_coarse,)
    std_noise  = noise_psds.std(axis=0, ddof=1)
    err_bar    = std_noise / np.sqrt(N)             # σ_noise / √N

    # --- 2. compute integer stride k ---------------------------------
    M_fine   = orig_psd.size
    M_coarse = noise_psds.shape[1]
    if (M_fine - 1) % (M_coarse - 1) != 0:
        raise ValueError(
            f"(len(orig_psd)-1)/(len(f_noise)-1) must be integer; "
            f"got ({M_fine}-1)/({M_coarse}-1)"
        )
    k = (M_fine - 1) // (M_coarse - 1)              # e.g. 4

    # --- 3. down-sample orig_psd -------------------------------------
    orig_on_noise = orig_psd[::k]                   # take every k-th
    if orig_on_noise.size != M_coarse:              # guard against round-off
        orig_on_noise = orig_on_noise[:M_coarse]

    # print
    print("size of noise array", orig_on_noise.size)
    print("noise array", orig_on_noise[:10])
    print("orig array", orig_psd[:10])

    # --- 4. normalised residual --------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        # residual = (orig_on_noise - mean_noise) / err_bar
        residual = (orig_on_noise - mean_noise) / orig_on_noise

    return residual