import numpy as np
import matplotlib.pyplot as plt
from scripts.psdNoise import psdTdiNoise
import random 

def compute_noise_psds(
    tdi_file_path,
    channel="X",
    n_trials=100,
    window="hann",
    nper=4096,
    noverlap=None,
    average="mean",
    scaling="density",
    seed_offset=0
):
    """
    Computes the true PSD and an ensemble of noise-only PSDs.

    Returns
    -------
    freq : np.ndarray
        Frequency array.
    psd_orig : np.ndarray
        The original PSD.
    noise_psds : list of np.ndarray
        List of noise-only PSD arrays (length n_trials).
    """
    # Compute the true PSD and initialize ensemble
    psd_orig, freq, _ = psdTdiNoise(
        tdi_file_path=tdi_file_path,
        channel=channel,
        window=window,
        nper=nper,
        noverlap=noverlap,
        average=average,
        scaling=scaling
    )
    noise_psds = []
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
        noise_psds.append(psd_noise)
    return freq, psd_orig, noise_psds

def plot_noise_psds(
    freq,
    psd_orig,
    noise_psds,
    n_to_plot=5,
    alpha=0.7
):
    """
    Randomly selects n_to_plot noise PSDs from the ensemble and plots them
    alongside the original PSD.
    """
    # Sample without replacement
    samples = random.sample(noise_psds, min(n_to_plot, len(noise_psds)))
    
    plt.figure(figsize=(8,5))
    for psd_noise in samples:
        plt.loglog(freq, psd_noise, alpha=alpha, lw=1)
    plt.loglog(freq, psd_orig, color='k', lw=2, label='Original PSD')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [1/Hz]")
    # set x-limits based on freq
    plt.xlim(freq[1], freq[-1])
    plt.title(f"{len(samples)} Random Noise PSDs vs. Original")
    plt.legend()
    plt.tight_layout()
    plt.show()