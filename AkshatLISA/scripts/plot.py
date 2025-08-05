import matplotlib.pyplot as plt
import scipy.signal as sig
from psd_estimation_methods.lpsd import lpsd
from psd_estimation_methods.wosa import wosa
import numpy as np
import random
from scipy.stats import chi2
from collections.abc import Iterable
from scipy.integrate import simpson
from scripts.medianFuncs import median_pdf, factor
from matplotlib.ticker import ScalarFormatter
from scripts.pdf import pdf                         
from scipy.integrate import quad

def plot_median_vs_mean_pdf(N, s=1.0, mode='scaled'):
    """
    Plot the WOSA-mean and WOSA-median PDFs for `N` segments and true PSD `s`.
    N : Number of segments 
    s : True one-sided PSD value used to generate the curves.
    mode : {'scaled', 'unscaled'}, default 'scaled'
        - 'scaled'   : stretch the median PDF so its mean equals `s`
        - 'unscaled' : leave the median PDF unchanged
    """
    df = 2 * N                   

    x = np.linspace(0, 50*s, 20000)
    pdf_mean = pdf(x, N, s, "mean")
      
    if mode == 'scaled':
        pdf_median = pdf(x, N, s, "median")
        mean_median = simpson(x * pdf_median, x=x)
        c          = s / mean_median                  
        pdf_median = median_pdf(x / c, N, s) / c
        mean_median = simpson(x * pdf_median, x=x) 
    else:
        pdf_median = pdf(x, N, s, "median")
        mean_median = simpson(x * pdf_median, x=x)

    second_moment_mean   = simpson(x**2 * pdf_mean,   x=x)
    std_mean             = np.sqrt(second_moment_mean   - s**2)
    second_moment_median = simpson(x**2 * pdf_median, x=x)
    std_median           = np.sqrt(second_moment_median - mean_median**2)
    
    plt.figure(figsize=(8, 4))
    plt.autoscale(enable=True, axis='y')
    
    line_mean, = plt.plot(
        x, pdf_mean,   lw=2, label=f'WOSA PDF (σ={std_mean:.3f})'
    )
    line_median, = plt.plot(
        x, pdf_median, lw=2, label=f'WOSA-Median PDF σ={std_median:.3f})'
    )
    
    plt.xlim(0, 3*s)
    plt.ylim(bottom=0)
    
    if mode == 'scaled':
        plt.axvline(
            s, ls='--', lw=2,
            label=f'Mean of PDF(s) = {s}'
        )
    else:
        plt.axvline(
            s, ls=':', lw=2,
            color=line_mean.get_color(),
            label=f'Mean mean = {s}'
        )
        plt.axvline(
            mean_median, ls=':', lw=2,
            color=line_median.get_color(),
            label=f'Mean median = {mean_median:.3f}'
        )
    
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.title(f'PDF for WOSA mean vs. median (PSD = {s}, N = {N})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_psd_noise_median_histogram_two_cross_two(
    noise_vals_list,
    freq_list,
    N_list,
    seg_counts,
    median_pdf_fn,
    main_title,
    n_bins=30,
    show_biased_hist=True
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(main_title, fontsize=16)
    
    for ax, psd_vals, freq, N, seg in zip(
        axes.flatten(),
        noise_vals_list,
        freq_list,
        N_list,
        seg_counts
    ):
        # 1) Histogram as raw counts
        counts, bins, _ = ax.hist(
            psd_vals,
            bins=n_bins,
            density=False,        # raw counts
            alpha=0.6,
            edgecolor='k'
        )
        total     = len(psd_vals)
        bin_width = bins[1] - bins[0]
        
        # 2) Build the two model‐median PDFs
        x0 = np.linspace(0, psd_vals.max()*1.1, 500)
        s  = psd_vals.mean()
        
        #   a) biased (uncentered)
        pdf0 = median_pdf_fn(x0, N, s)
        pdf0 /= np.trapz(pdf0, x0)
        
        #   b) aligned (centered at histogram mean)
        model_mean0 = np.trapz(x0 * pdf0, x0)
        scale = s / model_mean0
        x1    = x0 * scale
        pdf1  = pdf0 / scale
        
        # 3) Plot model curves **scaled to counts**
        if show_biased_hist:
            ax.plot(
                x0, pdf0 * total * bin_width,
                'b-', lw=2,
                label="Median PDF (biased)"
            )
        ax.plot(
            x1, pdf1 * total * bin_width,
            'r-', lw=2,
            label="Median PDF (centered at histogram mean)"
        )
        
        # 4) Vertical lines for the two means
        ax.axvline(
            model_mean0,
            color='b', ls='--', lw=2,
            label=f"True PSD estimate = {model_mean0:.2e}"
        )
        ax.axvline(
            s,
            color='r', ls='-.', lw=2,
            label=f"Histogram mean = {s:.2e}"
        )
        
        # 5) Disable scientific offset on y, and style
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis='y', style='plain')
        
        ax.set_xlim(left=0)
        ax.set_xlabel(f"PSD @ {freq:.3f} Hz")
        ax.set_ylabel("Count")
        ax.set_title(f"{seg} segments")
        ax.legend(fontsize='small')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot(data, t, f, psd, labels, title,
         show_time_series=True, f_lims=None,
        bigtitle=None, vlines=None, logpsd=False):
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
        ax0.set_xlabel(r"$Time \ [s]$")
        ax0.set_ylabel(title)
        ax0.grid(False)
        # ax0.set_title(r"$Time \ series$")

    # 2) PSD(s)
    ax1 = axes[-1]
    for psd_array, label in zip(psds, labels):
        if (logpsd):
            ax1.loglog(f, psd_array, 'o-', label=label)
            # ax1.set_yscale('log')
        else:
            ax1.loglog(f, psd_array, label=label)
    
    if f_lims is not None:
        ax1.set_xlim(f_lims[0], f_lims[1])
    
    if show_time_series and vlines is not None:
        for x in np.atleast_1d(vlines):
            ax0.axvline(x=t[x], ls=":", lw=1, color="k", alpha=0.6)

    ax1.set_xlabel(r"$Frequency \ [Hz]$")
    ax1.set_ylabel(r"$PSD \ [V^2/\sqrt{Hz}]$")
    ax1.legend()

    # ax.plot(f_logpsd, psd_logpsd, 'o-', label='log-PSD')   

    ax1.grid(False)

    fig.tight_layout()
    return fig, axes

def plot_segments(data, nperseg, overlap=0.5, t=None, fs=None, ax=None):
    """
    Plot concatenated overlapping segments from 1D data.
    
    Parameters
    ----------
    data : array-like
        Input signal of length N.
    nperseg : int
        Number of points in each segment.
    overlap : float or int, optional
        If float in (0,1), interpreted as fraction of nperseg.
        If int >= 1, interpreted as number of overlapping points.
        Default is 0.5 (50% overlap).
    t : array-like, optional
        Time vector corresponding to data. Used for x-axis scaling.
    fs : float, optional
        Sampling frequency (Hz). Used if `t` is not provided.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates a new figure and axes.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the plot.
    """
    # Determine overlap in points
    if isinstance(overlap, float) and 0 < overlap < 1:
        overlap_pts = int(overlap * nperseg)
    else:
        overlap_pts = int(overlap)
    step = nperseg - overlap_pts

    N = len(data)
    n_segments = (N - overlap_pts) // step
    if n_segments < 1:
        raise ValueError("Not enough data for at least one full segment.")

    # Extract and concatenate segments
    segments = [data[i*step : i*step + nperseg] for i in range(n_segments)]
    concatenated = np.concatenate(segments)

    # Build x-axis
    if t is not None:
        dt = t[1] - t[0]
        x = np.arange(len(concatenated)) * dt
        xlabel = "Time [s]"
    elif fs is not None:
        x = np.arange(len(concatenated)) / fs
        xlabel = "Time [s]"
    else:
        x = np.arange(len(concatenated))
        xlabel = "Sample index"

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    ax.plot(x, concatenated, lw=0.8)

    # Draw vertical lines at segment boundaries
    for i in range(1, n_segments):
        boundary = i * nperseg
        if t is not None:
            boundary = boundary * dt
        elif fs is not None:
            boundary = boundary / fs
        ax.axvline(boundary, ls=":", lw=1, color="k", alpha=0.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Overlapping segments (nperseg={nperseg}, overlap={overlap})")
    ax.set_xlim(0, x[-1])
    plt.tight_layout()
    return fig, ax

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
    """
    # pick up to n random noise realizations
    n_realizations = noise_psds.shape[0]
    idxs = random.sample(range(n_realizations), min(n, n_realizations))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(bigtitle, fontsize=16)

    # Plot the noise PSDs
    for i in idxs:
        ax.loglog(f_noise, noise_psds[i], alpha=alpha, lw=1)
    # Marker for chosen noise frequency
    # ax.axvline(chosen_f_noise, color='red', linestyle='--', linewidth=2,
    #            label=f"Noise freq: {chosen_f_noise:.5f} Hz")

    # Plot the original PSD
    ax.loglog(f_orig, orig_psd, color='k', lw=2, label="Original PSD")
    # Marker for chosen original frequency
    ax.axvline(chosen_f_orig, color='blue', linestyle='--', linewidth=2,
               label=f"Chosen Freq: {chosen_f_orig:.5f} Hz")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min(f_orig.min(), f_noise.min()),
                max(f_orig.max(), f_noise.max()))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    ax.legend()
    # ax.grid(True, which='both', ls='--', lw=0.5)
    ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
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

def plot_psd_noise_median_histogram(
    noise_vals,
    actual_freq,
    N,
    median_pdf_fn,
    n_bins=30,
    title="Noise PSD distribution"
):
    # 1) Empirical PDF (histogram, normalized to integrate to 1)
    plt.hist(
        noise_vals,
        bins=n_bins,
        density=True,
        alpha=0.6,
        edgecolor='k'
    )

    # 2) Build fine grid for the model PDF
    x0 = np.linspace(0, noise_vals.max() * 1.1, 500)

    # 2a) Uncentered PDF (blue, dashed)
    s = noise_vals.mean()
    pdf0 = median_pdf_fn(x0, N, s)
    pdf0 /= np.trapz(pdf0, x0)
    model_mean0 = np.trapz(x0 * pdf0, x0)

    # 2b) Aligned PDF (red, solid)
    scale = s / model_mean0
    x1 = x0 * scale
    pdf1 = pdf0 / scale

    plt.plot(
        x0, pdf0,
        'b-', lw=2,
        label="Median PDF (uncentered)"
    )
    plt.plot(
        x1, pdf1,
        'r-', lw=2,
        label="Median PDF (aligned)"
    )

    # 3) Vertical lines: 
    #    model‐mean (blue, dashed), histogram‐mean (red, dash‐dot)
    plt.axvline(
        model_mean0,
        color='b', lw=2, ls='--',
        label=f"Model mean = {model_mean0:.2e}"
    )
    plt.axvline(
        s,
        color='r', lw=2, ls='--',
        label=f"Histogram mean = {s:.2e}"
    )

    # 4) Decorations
    plt.xlabel(f"PSD @ {actual_freq:.3f} Hz [1/Hz]")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.xlim(left=0)
    plt.legend()
    plt.tight_layout()
    plt.show()

def draw_chi2(
    μ: float,
    df: int,
    show_percentile: float = 0,
    x_max: float = None,
    n_points: int = 500,
    title: str = "χ² Model"
):
    """
    Plot the χ² distribution model for a PSD estimator:
        pdf(x) = (df/μ) * χ²_pdf(df * x / μ; df)

    Optionally draw the vertical lines at the lower/upper bounds
    of the central `show_percentile`% interval.

    Parameters
    ----------
    μ : float
        The “true” PSD value around which the model is centered.
    df : int
        Degrees of freedom of the Welch estimator.
    show_percentile : float, optional
        Central percentile (0–100). If >0, draws the two edges
        of that CI; e.g. 95 → 2.5th & 97.5th percentiles.
    x_max : float, optional
        Maximum x-axis value. If None, defaults to 3·μ.
    n_points : int, optional
        Number of points to sample the PDF curve.
    title : str, optional
        Plot title.
    """
    # plotting range
    xmax = x_max if x_max is not None else μ * 3
    x = np.linspace(0, xmax, n_points)

    # χ²‐based PDF (normalized as a density)
    pdf = (df / μ) * chi2.pdf(df * x / μ, df)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, pdf, 'b-', lw=2, label=rf"$\chi^2_{{{df}}}$ PDF (μ={μ:.2e})")

    # central CI bounds
    if show_percentile > 0:
        # lower/upper percentile values
        lower_p = (100 - show_percentile) / 2
        upper_p = 100 - lower_p
        # inverse‐CDF for the scaled χ²: x_p = (μ/df) * χ²_ppf(p, df)
        x_low  = (μ / df) * chi2.ppf(lower_p  / 100.0, df)
        x_high = (μ / df) * chi2.ppf(upper_p  / 100.0, df)

        ax.axvline(
            x_low,
            color='g', lw=2, ls='--',
            label=f"{lower_p:.1f}th %ile = {x_low:.2e}"
        )
        ax.axvline(
            x_high,
            color='g', lw=2, ls='--',
            label=f"{upper_p:.1f}th %ile = {x_high:.2e}"
        )

    
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    # decorations
    ax.set_xlabel("PSD value [1/Hz]")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend()
    ax.grid(which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.show()

def plot_psd_noise_histogram(
        noise_vals,
        original_psd_value,     # ≈ true PSD at that bin (μ₁)
        actual_freq,
        df,                     # degrees of freedom
        n_bins=20,
        title="Noise PSD distribution",
        show_pdf=True,
        show_orig_pdf=True,
        show_percentile=0       # new: percentile (0–100) to plot, 0 = none
):
    # --- histogram as counts -----------------------------------------
    counts, bins, patches = plt.hist(
        noise_vals,
        bins=n_bins,
        density=False,       # raw counts
        alpha=0.6,
        edgecolor='k',
        label="Count"
    )

    μ1 = original_psd_value 
    μ2 = np.mean(noise_vals)       

    if show_pdf:
        # bin‐width for scaling the PDF curves to counts
        bin_width = bins[1] - bins[0]
        total = len(noise_vals)

        # x‐axis for PDF plotting
        x = np.linspace(0, noise_vals.max()*1.1, 500)

        # --- χ² PDF for original PSD model ---------------------------
        if show_orig_pdf:
            pdf1 = (df/μ1) * chi2.pdf(df * x / μ1, df)
            plt.plot(
                x,
                pdf1 * total * bin_width,
                'b-', lw=2,
                label=rf"$\chi^2_{{{df}}}$ model (orig. PSD)"
            )

            # mark the original PSD μ₁
            plt.axvline(
                μ1,
                color='b', lw=2, ls='--',
                label=f"Orig. PSD = {μ1:.2e}"
            )

        # --- χ² PDF for histogram‐average model -----------------------
        pdf2 = (df/μ2) * chi2.pdf(df * x / μ2, df)
        plt.plot(
            x,
            pdf2 * total * bin_width,
            'r-', lw=2,
            label=rf"$\chi^2_{{{df}}}$ model (hist. mean)"
        )

        # mark mean of histogram
        plt.axvline(
            μ2,
            color='r', lw=2, ls='--',
            label=f"Mean of histogram = {μ2:.2e}"
        )

    # if requested, mark the p-th percentile of the original‐PSD distribution
    if show_percentile:
        p = show_percentile / 100.0
        # solve F(x_p) = p for X ~ (μ1/df)*χ²_df  =>  x_p = (μ1/df)*chi2.ppf(p, df)
        x_p = (μ1/df) * chi2.ppf(p, df)
        plt.axvline(
            x_p,
            color='g', lw=2, ls=':',
            label=f"{show_percentile}th percentile = {x_p:.2e}"
        )

    # --- decorations ---------------------------------------------------
    plt.xlabel(f"PSD @ {actual_freq:.3f} Hz  [1/Hz]")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.xlim(left=0)
    plt.tight_layout()
    plt.show()

def plot_psd_noise_histogram_two_cross_two(
    noise_vals_list,   # list of 1D arrays, one per subplot
    orig_psd_list,     # list of floats: the “true” PSD at the chosen bin
    freq_list,         # list of floats: the frequency corresponding to that bin
    df_list,           # list of ints: degrees of freedom for each subplot
    seg_counts,        # list of ints: number of segments in each case
    main_title,        # str: the overall title above the 2×2 grid
    n_bins=30,
    show_biased_hist = True
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(main_title, fontsize=16)
    
    for ax, noise_vals, orig_psd, freq, df, seg in zip(
        axes.flatten(),
        noise_vals_list,
        orig_psd_list,
        freq_list,
        df_list,
        seg_counts
    ):
        # 1) raw‐count histogram
        counts, bins, _ = ax.hist(
            noise_vals,
            bins=n_bins,
            density=False,
            alpha=0.6,
            edgecolor='k'
        )
        # 2) scale χ²–PDF curves to histogram counts
        bin_width = bins[1] - bins[0]
        total = len(noise_vals)
        x = np.linspace(0, noise_vals.max()*1.1, 500)
        
        # model using original PSD
        if show_biased_hist:
            pdf1 = (df/orig_psd) * chi2.pdf(df * x / orig_psd, df)
            ax.plot(
                x,
                pdf1 * total * bin_width,
                'b-',
                lw=2,
                label=rf"$\chi^2_{{{df}}}$ (orig)"
            )
        
        # model using histogram mean
        mean_psd = noise_vals.mean()
        pdf2 = (df/mean_psd) * chi2.pdf(df * x / mean_psd, df)
        ax.plot(
            x,
            pdf2 * total * bin_width,
            'r-',
            lw=2,
            label=rf"$\chi^2_{{{df}}}$ (hist)"
        )
        
        # vertical markers
        ax.axvline(
            orig_psd,
            color='b',
            lw=2,
            ls='--',
            label=f"True PSD estimate = {orig_psd:.2e}"
        )

        
        ax.axvline(
            mean_psd,
            color='r',
            lw=2,
            ls='--',
            label=f"Histogram Mean = {mean_psd:.2e}"
        )
        
        # axis labels & title
        # ax.set_xlabel(f"PSD @ {freq:.3f} Hz [1/Hz]")
        ax.set_xlabel(f"PSD @ {freq:.3f} Hz")
        ax.set_ylabel("Count")
        ax.set_xlim(left=0)
        ax.set_title(f"{seg} segments")
        ax.legend(fontsize='small')
    
    # tighten around the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_median_vs_mean_pdf_two_cross_two(N_values=(1, 3, 10, 30), *, s=1.0, mode="scaled"):
    """
    Draw a 2×2 grid comparing the WOSA-mean and WOSA-median PDFs.

    Parameters
    ----------
    N_values : tuple/list of 4 ints, default (1, 3, 10, 30)
        Segment counts to use for the four sub-plots, ordered
        left-to-right, top-to-bottom.
    s        : float, default 1.0
        True one-sided PSD value used for every panel.
    mode     : {'scaled', 'unscaled'}, default 'scaled'
        • 'scaled'   – stretch the median PDF so its mean equals `s`  
        • 'unscaled' – leave the median PDF unchanged
    """
    if len(N_values) != 4:
        raise ValueError("N_values must contain exactly four integers (one per panel).")

    # ------------------------------------------------------------------
    # Helper: compute PDFs & stats for a single N
    # ------------------------------------------------------------------
    def _get_curves(n_seg):
        x = np.linspace(0, 50 * s, 20_000)
        pdf_mean = pdf(x, n_seg, s, "mean")        # relies on your existing `pdf`
        print(pdf_mean)

        if mode == "scaled":
            pdf_median = pdf(x, n_seg, s, "median")
            mean_med   = simpson(x * pdf_median, x=x)
            c          = s / mean_med
            pdf_median = median_pdf(x / c, n_seg, s) / c   # relies on your `median_pdf`
            mean_med   = simpson(x * pdf_median, x=x)
        else:
            pdf_median = pdf(x, n_seg, s, "median")
            mean_med   = simpson(x * pdf_median, x=x)

        σ_mean   = np.sqrt(simpson(x**2 * pdf_mean,   x=x) - s**2)
        σ_median = np.sqrt(simpson(x**2 * pdf_median, x=x) - mean_med**2)
        return x, pdf_mean, pdf_median, σ_mean, σ_median, mean_med

    # ------------------------------------------------------------------
    # Build figure & iterate over four panels
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()   # flatten for easy looping

    for ax, N in zip(axes, N_values):
        x, pdf_mean, pdf_median, σ_mean, σ_median, mean_med = _get_curves(N)

        line_mean,   = ax.plot(x, pdf_mean,   lw=2,
                               label=fr"WOSA PDF ($\sigma={σ_mean:.3f}$)")
        line_median, = ax.plot(x, pdf_median, lw=2,
                               label=fr"WOSA-median PDF ($\sigma={σ_median:.3f}$)")

        ax.set_xlim(0, 3 * s)
        ax.set_ylim(bottom=0)

        if mode == "scaled":
            ax.axvline(s, ls="--", lw=2, label=fr"Mean of PDFs = {s}")
        else:
            ax.axvline(s,        ls=":", lw=2, color=line_mean.get_color(),
                       label=fr"Mean₍mean₎ = {s}")
            ax.axvline(mean_med, ls=":", lw=2, color=line_median.get_color(),
                       label=fr"Mean₍median₎ = {mean_med:.3f}")

        ax.set_xlabel("x")
        ax.set_ylabel("PDF")
        ax.set_title(fr"PSD $={s}$, $N={N}$")
        ax.grid(True)
        ax.legend(fontsize="small")

    # ------------------------------------------------------------------
    # Overall layout
    # ------------------------------------------------------------------
    fig.suptitle(f"WOSA Mean vs. Median PDFs (PSD={s})",
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()