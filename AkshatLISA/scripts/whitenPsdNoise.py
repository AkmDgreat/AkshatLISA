import numpy as np
from numpy.fft import rfft, irfft
from scripts.psdNoise import time_noise_from_psd
from scripts.wosa import wosa

def whitened_time_noise_from_psd(psd, fs, nperseg, seed=None, eps=1e-24):
    """
    1. Draw a *coloured* Gaussian noise realisation whose PSD == `psd`.
    2. FFT-divide by √PSD  ⇒  unit (flat) one-sided PSD.
    
    Returns
    -------
    y : ndarray, length 2*(len(psd)-1)
        Real-valued, *whitened* time series (PSD ≈ 1 [1/Hz]).
    """
    # -- (1) coloured series --------------------------------------------------
    x = time_noise_from_psd(psd, fs, nperseg, seed=seed)

    # -- (2) whitening in the frequency domain --------------------------------
    N   = len(x)
    Xf  = rfft(x)
    W   = 1.0 / np.sqrt(psd + eps)          # |W(f)| = 1/√PSD
    Yf  = Xf * W                           # phase unchanged, magnitude flattened
    y   = irfft(Yf, n=N)

    return y

def n_whitened_time_noises_from_psd(psd, fs, nperseg, n=100, seed_offset=0):
    """
    Array of `n` whitened realisations, shape (n, nperseg).
    """
    ts = np.zeros((n, nperseg))
    for i in range(n):
        ts[i] = whitened_time_noise_from_psd(
                    psd, fs, nperseg,
                    seed = seed_offset + i
                )
    return ts

def n_whiten_noise_psds(time_noises, fs, nperseg, noverlap=None, window='hann'):
    """
    Same I/O as your `n_noise_psds`, but operates on *whitened* inputs.
    The returned PSDs should be ~1 everywhere (up to estimator variance).
    """
    n = time_noises.shape[0]
    noise_psds = np.zeros((n, nperseg // 2 + 1))

    for i in range(n):
        f, psd_i, _, nseg = wosa(x=time_noises[i],
                                 fs=fs,
                                 nperseg=nperseg,
                                 noverlap=noverlap,
                                 window=window)
        noise_psds[i] = psd_i

    return f, noise_psds, nseg

# ------------------------------------------------------------------ #
# (1)   Re-colour ONE whitened series back to the target PSD         #
# ------------------------------------------------------------------ #
def recolour_whitened_series(y_white, psd, eps=1e-24):
    """
    Multiply each frequency bin of a *whitened* time series by √PSD 
    so that the output has the original coloured spectrum.

    Parameters
    ----------
    y_white : ndarray
        Real-valued, whitened time series  (PSD ≈ 1).
    psd : array_like
        Target one-sided PSD (must align with rfft bins of y_white).
    eps : float, optional
        Floor to avoid multiplying by zero if psd contains exact zeros.

    Returns
    -------
    x_colour : ndarray
        Real-valued time series whose PSD ≈ psd.
    """
    N   = len(y_white)
    Yf  = rfft(y_white)
    G   = np.sqrt(psd + eps)           # inverse of whitening magnitude
    Xf  = Yf * G[:len(Yf)]             # apply re-colouring
    x_colour = irfft(Xf, n=N)
    return x_colour


# ------------------------------------------------------------------ #
# (2)   Re-colour *n* whitened realisations in one call              #
# ------------------------------------------------------------------ #
def n_recoloured_series(y_white_array, psd, eps=1e-24):
    """
    Parameters
    ----------
    y_white_array : ndarray, shape (n, N)
        Each row is a whitened realisation.
    psd : array_like
        Target PSD (same layout as in recolour_whitened_series).

    Returns
    -------
    ndarray, shape (n, N)
        Re-coloured realisations.
    """
    n, N = y_white_array.shape
    out  = np.zeros_like(y_white_array)

    for i in range(n):
        out[i] = recolour_whitened_series(y_white_array[i], psd, eps=eps)

    return out
