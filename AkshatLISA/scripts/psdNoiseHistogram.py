import numpy as np
import matplotlib.pyplot as plt
from scripts.psdNoise import psdTdiNoise

def plot_psd_noise_histogram(
    tdi_file_path,
    channel="X",
    target_freq=None,
    n_trials=100,
    window="hann",
    nper=4096,
    noverlap=None,
    average="mean",
    scaling="density",
    seed_offset=0
):
    """
    Computes the original PSD and then draws n_trials noise-psd realizations
    at the bin closest to target_freq.  Plots a histogram of the noise-psd values
    with the original PSD marked.

    Parameters
    ----------
    tdi_file_path : str
        Path to your .h5 TDI data file.
    channel : str
        Which TDI channel to read (default "X").
    target_freq : float
        Frequency (in Hz) at which to sample the PSD distribution.
    n_trials : int
        Number of noise realizations to draw.
    seed_offset : int
        Base for random seeds (you’ll get seeds = seed_offset + i).

    Other parameters are passed straight to psdTdiNoise().
    """
    # 1) get the “true” PSD
    psd_orig, freq, _ = psdTdiNoise(
        tdi_file_path=tdi_file_path,
        channel=channel,
        window=window,
        nper=nper,
        noverlap=noverlap,
        average=average,
        scaling=scaling,
    )

    # 2) identify the bin index nearest target_freq
    if target_freq is None:
        raise ValueError("You must specify target_freq")
    idx = np.argmin(np.abs(freq - target_freq))
    actual_freq = freq[idx]
    print(f"Sampling PSD at {actual_freq:.3f} Hz (closest to requested {target_freq:.3f} Hz)")

    # 3) draw n_trials noise‐PSD values at that bin
    noise_vals = np.empty(n_trials)
    for i in range(n_trials):
        # seed ensures each run is different if your function uses randomness
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

        
    return psd_orig, noise_vals, idx