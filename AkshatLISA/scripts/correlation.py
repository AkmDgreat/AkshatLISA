import numpy as np
import matplotlib.pyplot as plt
from scripts.psdNoise import psdTdiNoise

def compute_psd_noise_pair(
    tdi_file_path,
    channel="X",
    fraction=0.8,
    offset_bins=1,
    n_trials=100,
    window="hann",
    nper=4096,
    noverlap=None,
    average="mean",
    scaling="density",
    seed_offset=0
):
    """
    Computes parallel arrays of PSD noise values at two frequency bins.

    Parameters
    ----------
    tdi_file_path : str
        Path to your .h5 TDI data file.
    channel : str
        Which TDI channel to read (default "X").
    fraction : float
        Fractional position (0–1) for the first frequency bin.
    offset_bins : int
        Number of bins to offset for the second frequency.
    n_trials : int
        Number of noise realizations.
    seed_offset : int
        Base for random seeds (seed_offset + i).

    Returns
    -------
    noise1 : np.ndarray
        PSD noise values at the first frequency (length n_trials).
    noise2 : np.ndarray
        PSD noise values at the second frequency.
    freq1 : float
        Actual first frequency (Hz).
    freq2 : float
        Actual second frequency (Hz).
    """
    # 1) Compute "true" PSD + frequency vector
    psd_orig, freq, _ = psdTdiNoise(
        tdi_file_path=tdi_file_path,
        channel=channel,
        window=window,
        nper=nper,
        noverlap=noverlap,
        average=average,
        scaling=scaling
    )
    n_bins = len(freq)
    idx1 = int(fraction * (n_bins - 1))
    idx2 = idx1 + offset_bins
    if idx2 < 0 or idx2 >= n_bins:
        raise IndexError("Offset bin out of range")

    freq1, freq2 = freq[idx1], freq[idx2]
    
    # 2) Generate noise PSDs
    noise1 = np.empty(n_trials)
    noise2 = np.empty(n_trials)
    for i in range(n_trials):
        _, _, psd_noise = psdTdiNoise(
            tdi_file_path=tdi_file_path,
            channel=channel,
            window=window,
            nper=nper,
            noverlap=noverlap,
            average=average,
            scaling=scaling,
            seed=seed_offset + i
        )
        noise1[i] = psd_noise[idx1]
        noise2[i] = psd_noise[idx2]

    return noise1, noise2, freq1, freq2

def plot_psd_noise_correlation(
    noise1,
    noise2,
    freq1,
    freq2,
    title="PSD Noise Correlation"
):
    """
    Scatter‐plots two PSD noise arrays to visualize correlation.

    Parameters
    ----------
    noise1 : np.ndarray
        PSD noise values at frequency1.
    noise2 : np.ndarray
        PSD noise values at frequency2.
    freq1 : float
        Frequency1 in Hz.
    freq2 : float
        Frequency2 in Hz.
    """
    plt.figure(figsize=(6,6))
    plt.scatter(noise1, noise2, alpha=0.7, edgecolor='k')
    # add y=x reference
    mn = min(noise1.min(), noise2.min())
    mx = max(noise1.max(), noise2.max())
    plt.plot([mn, mx], [mn, mx], linestyle='--', label='y = x')
    plt.xlabel(f"PSD noise @ {freq1:.5f} Hz")
    plt.ylabel(f"PSD noise @ {freq2:.5f} Hz")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()