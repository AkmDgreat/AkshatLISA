import numpy as np
import matplotlib.pyplot as plt
from scripts.psdNoise import psdTdiNoise
import numpy as np

def compute_psd_noise_pair(
    tdi_file_path,
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
    Draw PSD-noise samples at two frequency bins and compute

        ⟨(X - X_true)(Y - Y_true)⟩                (covariance)
        ⟨(X - X_true)(Y - Y_true)⟩ / (sigma_X sigma_Y)     (correlation)

    where X_true and Y_true are the *deterministic* PSD values returned
    by `psdTdiNoise`.

    Returns
    -------
    cov_xy : float
        Population covariance about the true means.
    rho_xy : float
        Pearson correlation coefficient (about the true means).
    noise1, noise2 : np.ndarray
        The two sample vectors (length n_trials).
    freq1, freq2 : float
        Frequencies of the two bins (Hz).
    x_true, y_true : float
        The “true” PSD values at the two bins.
    """
    # ---------- first call: one deterministic PSD --------------------
    

    psd_true, freq, _ = psdTdiNoise(
        tdi_file_path=tdi_file_path,
        channel=channel,
        window=window,
        nper=nper,
        noverlap=noverlap,
        average=average,
        scaling=scaling,
        seed=None            
    )
    n_bins = len(freq)
    idx1 = int(fraction * (n_bins - 1))
    idx2 = idx1 + offset_bins
    if idx2 < 0 or idx2 >= n_bins:
        raise IndexError("Offset bin out of range")
    x_true = psd_true[idx1]
    y_true = psd_true[idx2]
    freq1, freq2 = freq[idx1], freq[idx2]

    # ---------- second step: stochastic noise realisations ----------
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
            seed=seed_offset + i      # reproducible ensemble
        )
        noise1[i] = psd_noise[idx1]
        noise2[i] = psd_noise[idx2]

    # ---------- covariance & correlation about *true* means ---------
    cov_xy = np.mean((noise1 - x_true) * (noise2 - y_true))
    sigma_x = np.sqrt(np.mean((noise1 - x_true) ** 2))
    sigma_y = np.sqrt(np.mean((noise2 - y_true) ** 2))
    rho_xy = cov_xy / (sigma_x * sigma_y)

    return noise1, noise2, freq1, freq2, x_true, y_true, cov_xy, rho_xy

def plot_psd_noise_correlation(
    noise1,
    noise2,
    freq1,
    freq2,
    rho_xy=None,         # <- new (optional) argument
    title="PSD Noise Correlation"
):
    """
    Scatter-plot two PSD-noise arrays and, if supplied, display the
    Pearson correlation coefficient ρ in a text box.

    Parameters
    ----------
    noise1, noise2 : np.ndarray
        PSD noise values at two frequency bins.
    freq1, freq2   : float
        Corresponding frequencies (Hz).
    rho_xy         : float or None
        Correlation coefficient to display.  If None, nothing is shown.
    title          : str
        Plot title.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(noise1, noise2, alpha=0.7, edgecolor="k")

    # 45-degree reference line
    mn = min(noise1.min(), noise2.min())
    mx = max(noise1.max(), noise2.max())
    plt.plot([mn, mx], [mn, mx], ls="--", label="y = x")

    # annotate ρ if provided
    if rho_xy is not None:
        text = fr"$\rho = {rho_xy:.3f}$"
        plt.gca().text(
            0.05, 0.95, text,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    plt.xlabel(f"PSD noise @ {freq1:.5f} Hz")
    plt.ylabel(f"PSD noise @ {freq2:.5f} Hz")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------
def compute_rho_vs_freq(
    tdi_file_path,
    channel="X",
    fraction=0.8,          # pick the reference bin (f₁)
    n_trials=100,
    window="hann",
    nper=4096,
    noverlap=None,
    average="mean",
    scaling="density",
    seed_offset=0,
):
    """
    Fix one frequency bin f₁ (chosen by `fraction`) and return ρ(f, f₁)
    for every other bin f.

    Returns
    -------
    freq : (n_bins,) ndarray
        Frequency grid (Hz).
    rho  : (n_bins,) ndarray
        Correlation coefficient between PSD-noise at f and at f₁.
    idx_ref : int
        Index of the reference bin (so rho[idx_ref] ≃ 1).
    """
    # 1) deterministic PSD → “true” means
    psd_true, freq, _ = psdTdiNoise(
        tdi_file_path=tdi_file_path,
        channel=channel,
        window=window,
        nper=nper,
        noverlap=noverlap,
        average=average,
        scaling=scaling,
        seed=None,                  # deterministic
    )
    n_bins  = len(freq)
    idx_ref = int(fraction * (n_bins - 1))
    x_true  = psd_true[idx_ref]     # μ₁
    # 2) draw a matrix of noise realisations  (n_trials × n_bins)
    noise_mat = np.empty((n_trials, n_bins))
    for i in range(n_trials):
        _, _, psd_noise = psdTdiNoise(
            tdi_file_path=tdi_file_path,
            channel=channel,
            window=window,
            nper=nper,
            noverlap=noverlap,
            average=average,
            scaling=scaling,
            seed=seed_offset + i,
        )
        noise_mat[i] = psd_noise
    # 3) deviations from the true means
    dP = noise_mat - psd_true        # shape (trials, bins)
    dP_ref = dP[:, idx_ref]          # shape (trials,)
    # 4) population σ for every bin
    sigma = np.sqrt(np.mean(dP**2, axis=0))
    sigma_ref = sigma[idx_ref]
    # 5) covariance with the reference column, then ρ
    cov = np.mean(dP_ref[:, None] * dP, axis=0)   # shape (bins,)
    rho = cov / (sigma_ref * sigma)
    # rho[idx_ref] = 1.0                           # force exact self-corr
    return freq, rho, idx_ref

# ------------------------------------------------------------------
def plot_rho_vs_freq(freq, rho, idx_ref, title="ρ(f, f₁) vs frequency"):
    plt.figure(figsize=(7.5, 4))
    plt.plot(freq, np.abs(rho), lw=1.4)
    # plt.axvline(freq[idx_ref], color="crimson", ls="--",
    #             label=f"reference f₁ = {freq[idx_ref]:.5f} Hz")
    plt.xlim(freq[0], freq[-1])     
    plt.ylim(0, 1)       
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Correlation  $\rho(f,f_1)$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
# ------------------------------------------------------------------