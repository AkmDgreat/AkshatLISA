import matplotlib.pyplot as plt
import scipy.signal as sig
from scripts.lpsd import lpsd
from scripts.wosa import wosa
import numpy as np
import random
from scipy.stats import chi2

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
def plot(data, t, fs, f, psd, nper, title, bigtitle="PSD graphs"):
    # 1×3 subplots, make it wide enough
    fig, axes = plt.subplots(1, 2, figsize=(20, 4))
    fig.suptitle(bigtitle, fontsize=16)

    # prep common quantities
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
    ax.loglog(f, psd)
    ax.set_xlim(f_min, f_max)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    ax.set_title("WOSA (log-log)")
    fig.tight_layout()

def plot_orig_and_noise_psds(
    f_orig,
    orig_psd,
    chosen_f_orig,
    f_noise,
    noise_psds,
    chosen_f_noise,
    n=5,
    alpha=0.3,
    bigtitle="PSD Comparison"
):
    """
    Plot the original PSD alongside a vertical marker at `chosen_f_orig`,
    and on the right plot show `n` random noise PSD realizations with
    their vertical marker at `chosen_f_noise`.

    Parameters
    ----------
    f_orig : array_like
        Frequency bins for the original PSD.
    orig_psd : array_like
        Original PSD values.
    chosen_f_orig : float
        Frequency (Hz) at which to draw a vertical line on the original PSD.
    f_noise : array_like
        Frequency bins for the noise PSDs.
    noise_psds : array_like, shape (n_realizations, len(f_noise))
        PSD realizations for the noise.
    chosen_f_noise : float
        Frequency (Hz) at which to draw a vertical line on the noise PSDs.
    n : int, optional
        How many of the noise realizations to plot. Default is 5.
    alpha : float, optional
        Line transparency for the noise PSDs. Default is 0.3.
    bigtitle : str, optional
        Supertitle for the entire figure.
    """
    # pick up to n random rows
    n_realizations = noise_psds.shape[0]
    idxs = random.sample(range(n_realizations), min(n, n_realizations))

    fig, axes = plt.subplots(1, 2, figsize=(20, 4))
    fig.suptitle(bigtitle, fontsize=16)

    # Left: original PSD
    ax0 = axes[0]
    ax0.loglog(f_orig, orig_psd, label="Original PSD")
    ax0.set_xlim(f_orig[0], f_orig[-1])
    ax0.set_xlabel("Frequency [Hz]")
    ax0.set_ylabel("PSD [1/Hz]")
    ax0.set_title("Original PSD (log-log)")
    ax0.axvline(chosen_f_orig, color='red', linewidth=2,
                label=f"Chosen freq: {chosen_f_orig:.5f} Hz")
    ax0.legend()

    # Right: noise PSDs
    ax1 = axes[1]
    for i in idxs:
        ax1.loglog(f_noise, noise_psds[i], alpha=alpha, lw=1)
    # ax1.loglog(f_orig, orig_psd, color='k', lw=2, label="Original PSD")
    ax1.set_xlim(f_noise[0], f_noise[-1])
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("PSD [1/Hz]")
    ax1.set_title(f"{len(idxs)} Noise PSDs vs. Original")
    ax1.axvline(chosen_f_noise, color='red', linewidth=2,
                label=f"Chosen freq: {chosen_f_noise:.5f} Hz")
    ax1.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

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



from scipy.stats import chi2




# Plot

# plt.title(f"Chi-Square Distribution PDF (df={df})")
# plt.xlabel("x")
# plt.ylabel("Probability Density")
# plt.grid(True)
# plt.show()

# def plot_psd_noise_histogram(
#     noise_vals,
#     original_psd_value,
#     actual_freq,
#     n_bins=20,
#     title="Noise PSD distribution"
# ):
#     """
#     Plots a histogram of noise-only PSD values with the original PSD marked.

#     Parameters
#     ----------
#     noise_vals : np.ndarray
#         Noise-only PSD values.
#     original_psd_value : float
#         The original PSD value to overlay.
#     actual_freq : float
#         Frequency in Hz for annotation.
#     n_bins : int
#         Number of histogram bins.
#     """
#     plt.figure()
#     plt.hist(noise_vals, bins=n_bins, edgecolor='k')
#     # noise_vals * (actual_freq / 14)
#     plt.axvline(original_psd_value, color='r', lw=2, label="Original PSD")
#     plt.xlabel(f"PSD @ {actual_freq:.3f} Hz [1/Hz]")
#     plt.ylabel("Count")
#     plt.title(title)

#     df = 14
#     # Values for x
#     x = np.linspace(0, 5e-39, 500)
#     pdf = chi2.pdf(x / (original_psd_value / df), df)
#     plt.plot(x, pdf*1000*1.25)

#     plt.show()

def plot_psd_noise_histogram(
        noise_vals,
        original_psd_value,     # mean (≈ true PSD at that bin)
        actual_freq,
        df,  # degrees of freedom
        n_bins=20,            
        title="Noise PSD distribution"
):
    # --- histogram as *probability density* ---------------------------
    plt.figure()
    plt.hist(noise_vals,
             bins=n_bins,
             density=True,              
             alpha=0.6,
             edgecolor='k',
             label="Empirical density")

    # --- χ² PDF for Welch estimator -----------------------------------
    μ  = original_psd_value                 # E[ P̂ ]  (theory or sample mean)
    x  = np.linspace(0, noise_vals.max()*1.1, 500)
    pdf = (df/μ) * chi2.pdf(df * x / μ, df)   # rescaled χ²
    plt.plot(x, pdf, 'r-', lw=2, label=rf"$\chi^2_{{{df}}}$ PDF")

    # --- decorations ---------------------------------------------------
    plt.axvline(original_psd_value, color='m', lw=2, ls='--', label="Original PSD")
    plt.xlabel(f"PSD @ {actual_freq:.3f} Hz  [1/Hz]")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
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
