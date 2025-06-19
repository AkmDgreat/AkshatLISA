import matplotlib.pyplot as plt
import scipy.signal as sig
from scripts.lpsd import lpsd
from scripts.wosa import wosa
import numpy as np
import random
from scipy.stats import chi2
from collections.abc import Iterable

def plot_residual_histogram(residual: np.ndarray,
                            bins: int | str = "auto",
                            **hist_kwargs) -> None:
    """
    Draw a histogram of PSD residuals.

    Parameters
    ----------
    residual : ndarray
        Array returned by `normalized_psd_residual`.
    bins : int | str, optional
        Number of bins or any valid `numpy.histogram_bin_edges` argument
        (default "auto").
    **hist_kwargs
        Extra keyword arguments forwarded to `plt.hist`
        (e.g. alpha=0.8, color="tab:blue").

    Notes
    -----
    The function creates its own figure/axis and immediately shows
    the plot; it does not return anything.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(residual,
             bins=bins,
             histtype="stepfilled",
             edgecolor="k",
             **hist_kwargs)

    plt.xlabel(r"Normalised residual $(\mathrm{PSD}_{\rm orig}-\langle\mathrm{PSD}_{\rm noise}\rangle)\;/\;(\sigma/\sqrt{N})$")
    plt.ylabel("Count")
    plt.title("Histogram of PSD residuals")
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.show()

def plot_psd_bias(freq: np.ndarray,
                  residual: np.ndarray,
                  *,
                  mark_freq: float | Iterable[float] | None = None,
                  title: str = r"$(\mathrm{PSD}_{\rm orig}-\langle\mathrm{PSD}_{\rm noise}\rangle)\;/\;\sigma_{\rm SE}$"
                 ) -> None:
    """
    Plot the normalised PSD residual versus frequency (log-x).

    Parameters
    ----------
    freq       : array_like, shape (n_freq,)
        Frequency bins (must all be > 0 for log scale).
    residual   : array_like, shape (n_freq,)
        Output from `normalized_psd_residual`.
    mark_freq  : float or iterable of float, optional
        If provided, draw dotted vertical line(s) at those frequency/ies.
    title      : str
        Plot title (LaTeX allowed).
    """
    freq = np.asarray(freq)
    residual = np.asarray(residual)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xscale("log")

    ax.plot(freq, residual, lw=1)
    ax.axhline(0, ls="--", alpha=0.6)
    for lvl in (1, 2):
        ax.axhline( lvl, ls=":", alpha=0.4)
        ax.axhline(-lvl, ls=":", alpha=0.4)

    # ── optional vertical markers ────────────────────────────────────
    if mark_freq is not None:
        # accept a single float or an iterable
        if isinstance(mark_freq, (int, float)):
            mark_freq = [mark_freq]
        elif not isinstance(mark_freq, Iterable):
            raise TypeError("mark_freq must be a number or an iterable of numbers")

        for f in mark_freq:
            if f <= 0:
                raise ValueError("mark_freq must be positive for a log axis")
            ax.axvline(f, ls=":", lw=1.2, color="red", alpha=0.7)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Normalised residual")
    ax.set_title(title)
    ax.set_xlim(freq[freq > 0][0], freq[-1])
    ax.set_ylim(-1, 1)

    plt.show()

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

def plot(data, t, f, psd, labels, title,
         show_time_series=True,
         nper=None, fs=None, bigtitle=None, vlines=None, logpsd=False):
    """
    Plot a time-series and one or more PSD estimates side by side,
    or just the PSD if show_time_series=False.
    """
    # wrap single inputs into lists
    if not isinstance(psd, (list, tuple)):
        psds = [psd]
    else:
        psds = psd

    if not isinstance(labels, (list, tuple)):
        labels = [labels]

    if len(psds) != len(labels):
        raise ValueError("`psd` and `labels` must have the same length")

    # choose layout
    ncols = 2 if show_time_series else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4))

    if bigtitle is not None:
        fig.suptitle(bigtitle, fontsize=16)

    # if only one axis, wrap it in a list so indexing below still works
    if ncols == 1:
        axes = [axes]

    # 1) time-domain (optional)
    if show_time_series:
        ax0 = axes[0]
        ax0.plot(t, data)
        ax0.set_xlim(t[0], t[-1])
        ax0.set_xlabel("Time [s]")
        ax0.set_ylabel(title)
        ax0.set_title("Time series")

    # 2) log–log PSD(s)
    ax1 = axes[-1]
    for psd_array, label in zip(psds, labels):
        if (logpsd):
            ax1.plot(f, psd_array, label=label)
            ax1.set_yscale('log')
        else:
            ax1.loglog(f, psd_array, label=label)

    # uncomment to remove limits 
    # if (nper is not None) and (fs is not None):
    #     ax1.set_xlim(fs/nper, fs/2.0)
    
    if vlines is not None:
        # make whatever the user passed iterable
        for v in np.atleast_1d(vlines):
            ax1.axvline(v, ls=":", lw=1, color="k", alpha=0.6)

    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("PSD (LISA)")
    # ax1.set_title("WOSA (log–log)")
    ax1.legend()

    ax0.grid(False)
    ax1.grid(False)

    fig.tight_layout()
    return fig, axes

# def draw_segments(t, data, nperseg, noverlap):
#     N = len(data)
#     plt.figure()
#     plt.plot(t, data)
#     step = nperseg - noverlap
#     starts = np.arange(0, N, step)
#     for start in starts:
#         end = start + nperseg
#         plt.axvline(start, color='C0', linestyle='--')
#         plt.axvline(min(end, N-1), color='C1', linestyle='--')
#     plt.show()

def draw_segments(t, data, nperseg, noverlap):
    N = len(data)
    step = nperseg - noverlap
    starts = np.arange(0, N, step)

    plt.figure()
    plt.plot(t, data)
    plt.xlim(t[0], t[-1])

    for start in starts:
        # always draw the start line
        plt.axvline(t[start], color='C1', linestyle='--')
        # only draw an end line if it falls inside the data
        end = start + nperseg
        if end < N:
            plt.axvline(t[end], color='C0', linestyle='--')

    plt.show()


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

from scripts.medianFuncs import median_exp_pdf

def plot_median_histogram(noise_vals, N, scale, labels,
                          title="Histogram", n_bins=30, cl=None, conf=None):

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(noise_vals, bins=n_bins, density=True,
            alpha=0.6, edgecolor='k', label='Empirical PDF')
    # ax.hist(noise_vals, bins=n_bins,
    #         alpha=0.6, edgecolor='k', label='Empirical PDF')

    x = np.linspace(0, noise_vals.max() * 1.1, 500)
    scales = np.atleast_1d(scale)

    # Handle labels
    labels = np.atleast_1d(labels)
    if labels.size != scales.size:
        raise ValueError("`labels` and `scale` must have the same length")
    plot_labels = labels

    # Plot one curve per scale
    for s, lbl in zip(scales, plot_labels):
        pdf = median_exp_pdf(x, N, scale=s)
        # print(pdf)
        # ax.axvline(np.mean(pdf), linestyle='--', label="Mean of curve")
        # print(f"PDF mean: {np.mean(pdf)}")

        # 1.  Normalise (robust against finite grid error)
        pdf /= np.trapz(pdf, x)         

        # 2.  Numerical expectation  μ = ∫ x f(x) dx
        mu = np.trapz(x * pdf, x)        # or: dx = x[1]-x[0];  mu = np.sum(x*pdf)*dx

        # 3.  Plot the mean of the theoretical curve
        ax.axvline(mu,
                linestyle='-.',
                color='tab:orange',
                label=f'Mean of {lbl}')
                
        ax.plot(x, pdf, lw=2, label=lbl)
    
    ax.axvline(np.mean(noise_vals), linestyle='--', label="Mean of histogram")
    print(f"histogram: {noise_vals}")
    # Add CI lines if given
    if cl is not None and conf is not None:
        pct = int(conf * 100)
        ax.axvline(cl[0], linestyle='--', label=f'Lower {pct}% CI')
        ax.axvline(cl[1], linestyle='--', label=f'Upper {pct}% CI')

    ax.set_xlabel("PSD")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_psd_noise_histogram(
        noise_vals,
        original_psd_value,     # ≈ true PSD at that bin (μ)
        actual_freq,
        df,                     # degrees of freedom
        n_bins=20,
        title="Noise PSD distribution",
):
    # --- histogram as *probability density* ---------------------------
    plt.figure()
    plt.hist(noise_vals,
             bins=n_bins,
             density=True,
             alpha=0.6,
             edgecolor='k',
             label="probability density")

    # --- χ² PDF for Welch estimator -----------------------------------
    x  = np.linspace(0, noise_vals.max()*1.1, 500)

    μ1  = original_psd_value                 
    pdf = (df/μ1) * chi2.pdf(df * x / μ1, df)   # rescaled χ²
    plt.plot(x, pdf, 'b-', lw=2, label=rf"$\chi^2_{{{df}}}$ original PSD")

    μ2  = np.mean(noise_vals)
    pdf = (df/μ2) * chi2.pdf(df * x / μ2, df)   # rescaled χ²
    plt.plot(x, pdf, 'r-', lw=2, label=rf"$\chi^2_{{{df}}}$ Histogram average")

    # 1) “Original PSD” at this frequency bin (already in the code)
    plt.axvline(original_psd_value,
                color='b', lw=2, ls='--',
                label=f"Original PSD = {original_psd_value:.3e}")

    # 2) Mean of the the noise_vals (the mean of histogram)
    mean_of_psd = np.mean(noise_vals)
    plt.axvline(mean_of_psd,
                color='r', lw=2, ls='--',
                label=f"Mean of histogram = {mean_of_psd:.3e}")

    # --- decorations ---------------------------------------------------
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
