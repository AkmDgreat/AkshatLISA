import matplotlib.pyplot as plt
import scipy.signal as sig
from scripts.lpsd import lpsd
from scripts.wosa import wosa
import numpy as np

def plotLigoPsd(data,
                t,
                dt,
                nper,
                title,
                bigtitle="PSD graphs",
                window='hann',
                scaling='density',
                average="mean",
                noverlap=None):
    """Time series + PSD with 5th/95th-percentile envelopes."""
    
    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))   # 2 columns are enough
    fig.suptitle(bigtitle, fontsize=16)
    
    # ------------------------------------------------------------------- setup
    fs    = 1.0 / dt
    f_min = 1.0 / (nper * dt)
    f_max = fs / 2.0
    
    # ----------------------------------------------------------------- panel 0
    ax = axes[0]
    ax.plot(t, data)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(title)
    ax.set_title("Time series")
    
    # --------------------------------------------------------------- run WOSA
    f_w, psd_w, P_stack = wosa(          # <<< grab the per-segment stack
        data,
        fs=fs,
        window=window,
        nperseg=nper,
        noverlap=noverlap,
        scaling=scaling,
        average=average,
    )
    
    # ---------------------------------------------------- 5th / 95th percentiles
    psd_p5  = np.percentile(P_stack,  5, axis=0)   # (nfft//2+1,)
    psd_p95 = np.percentile(P_stack, 95, axis=0)
    
    # ----------------------------------------------------------------- panel 1
    ax = axes[1]
    ax.loglog(f_w, psd_w,  label='mean/median PSD')
    ax.loglog(f_w, psd_p5,  ls='-', color='tab:gray', label='5th percentile')
    ax.loglog(f_w, psd_p95, ls='-', color='tab:gray', label='95th percentile')
    
    ax.set_xlim(f_min, f_max)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    ax.set_title("WOSA with 5 % / 95 % envelopes")
    ax.legend()
    
    plt.tight_layout()
    plt.show()                         

# plot the figures horizontally
def plot(data, t, fs, nper, title, bigtitle="PSD graphs"):
    # 1×3 subplots, make it wide enough
    fig, axes = plt.subplots(1, 2, figsize=(20, 4))
    fig.suptitle(bigtitle, fontsize=16)

    # prep common quantities
    fs    = 1.0 / dt
    f_min = fs / nper
    f_max = fs / 2.0

    # 1) time-domain
    ax = axes[0]
    ax.plot(t, data)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(title)
    ax.set_title("Time series")

    # 2) WOSA PSD (log–log)
    ax = axes[1]
    ax.loglog(f_w, psd_w)
    ax.set_xlim(f_min, f_max)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    ax.set_title("WOSA (log-log)")
    fig.tight_layout()


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
    plt.xlim(freq[1], freq[-1])
    plt.title(f"{len(samples)} Random Noise PSDs vs. Original")
    plt.legend()
    plt.tight_layout()
    plt.show()


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

def ligoHistogram(P_stack, Pxx, seg_idx, bins='auto'):
    """
    Histogram all PSD values in one segment and overlay the
    overall (Welch/WOSA-averaged) PSD as a red reference.

    Parameters
    ----------
    P_stack : (nseg, nfft//2+1) array
        Per-segment one-sided, scaled PSDs returned by wosa().
    Pxx     : (nfft//2+1,) array
        Segment-averaged PSD returned by wosa().
    seg_idx : int
        Which segment to inspect (0-based; 4 → the fifth segment).
    bins    : int or str, optional
        Histogram bin spec passed straight to `plt.hist`.
    """

    # -------- sanity checks --------------------------------------------------
    if seg_idx < 0 or seg_idx >= P_stack.shape[0]:
        raise IndexError("seg_idx must be in [0, nseg-1].")

    vals = P_stack[seg_idx]          # length nfft//2 + 1

    # -------- histogram ------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=bins, alpha=0.75, edgecolor='k')
    plt.xlabel("PSD value")
    plt.ylabel("Count")
    plt.title(f"Segment {seg_idx+1} – distribution of {vals.size} PSD bins")

    # -------- overlay reference ---------------------------------------------
    # Option A: a single vertical line at the *mean* PSD
    plt.axvline(Pxx.mean(), color='red', linewidth=2, label="mean Pxx")

    # Option B (comment the line above, uncomment below) to show every Pxx bin
    # plt.scatter(Pxx, [0]*len(Pxx), color='red', marker='|', s=100, label='Pxx bins')

    plt.legend()
    plt.tight_layout()
    plt.show()
