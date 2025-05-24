import numpy as np
import matplotlib.pyplot as plt
from scripts.psdNoise import psdTdiNoise
import random 

def compute_psd_noise_distribution(
    tdi_file_path,
    channel="X",
    fraction=0.5,
    n_trials=100,
    window="hann",
    nper=4096,
    noverlap=None,
    average="mean",
    scaling="density",
    seed_offset=0
):
    """
    Computes the original PSD and an array of noise-only PSD values at the frequency bin
    specified by a fractional position in the frequency series.

    Parameters
    ----------
    tdi_file_path : str
        Path to the .h5 TDI data file.
    channel : str
        TDI channel to read (default "X").
    fraction : float
        Fractional index (0 to 1) into the frequency array (e.g., 0.9 → 90% of the way).
    n_trials : int
        Number of noise realizations to generate.
    seed_offset : int
        Base for random seeds (seeds = seed_offset + i).

    Other parameters are passed to psdTdiNoise().
    
    Returns
    -------
    noise_vals : np.ndarray
        Array of length n_trials of noise-only PSD values at the selected bin.
    original_value : float
        The "true" PSD value at that bin.
    actual_freq : float
        The frequency (in Hz) at that bin.
    """
    # 1) Compute the true PSD and frequency array
    psd_orig, freq, _ = psdTdiNoise(
        tdi_file_path=tdi_file_path,
        channel=channel,
        window=window,
        nper=nper,
        noverlap=noverlap,
        average=average,
        scaling=scaling
    )
    # 2) Determine the bin index by fraction
    n_bins = len(freq)
    if not (0 <= fraction <= 1):
        raise ValueError("fraction must be between 0 and 1")
    idx = min(int(fraction * (n_bins - 1)), n_bins - 1)
    actual_freq = freq[idx]
    original_value = psd_orig[idx]

    # 3) Generate noise-only PSD samples
    noise_vals = np.empty(n_trials)
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
        noise_vals[i] = psd_noise[idx]

    return noise_vals, original_value, actual_freq

def plot_psd_noise_histogram(
    noise_vals,
    original_value,
    actual_freq,
    n_bins=20,
    title="Noise PSD distribution"
):
    """
    Plots a histogram of noise-only PSD values with the original PSD marked.

    Parameters
    ----------
    noise_vals : np.ndarray
        Noise-only PSD values.
    original_value : float
        The original PSD value to overlay.
    actual_freq : float
        Frequency in Hz for annotation.
    n_bins : int
        Number of histogram bins.
    """
    plt.figure()
    plt.hist(noise_vals, bins=n_bins, edgecolor='k')
    plt.axvline(original_value, color='r', lw=2, label="Original PSD")
    plt.xlabel(f"PSD @ {actual_freq:.3f} Hz [1/Hz]")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()