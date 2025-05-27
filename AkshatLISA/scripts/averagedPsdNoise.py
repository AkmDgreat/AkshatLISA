import numpy as np
import scipy.signal
from gwpy.timeseries import TimeSeriesDict
from scripts.wosa import wosa          # your existing wrapper

# ---------------------------------------------------------------------
# helper: ONE time-domain noise segment (length = nper) ----------------
#         uses YOUR σ-formula, no change to the maths
# ---------------------------------------------------------------------
def _noise_segment_from_psd(psd_seg, fs, nper, seed=None):
    """
    Fabricates one real-valued segment of length `nper` whose expected
    PSD equals `psd_seg` (one-sided, length nper//2 + 1).
    """
    # ---------- frequency-domain draw (your code verbatim) ------------
    sigma = np.sqrt(psd_seg * fs * nper / 4)
    sigma[0]  = np.sqrt(psd_seg[0]  * fs * nper / 2)     # DC
    if nper % 2 == 0:
        sigma[-1] = np.sqrt(psd_seg[-1] * fs * nper / 2) # Nyquist

    if seed is not None:
        np.random.seed(seed)

    not_zero   = (sigma != 0)
    sigma_red  = sigma[not_zero]
    noise_re   = np.random.normal(0, sigma_red)
    noise_co   = np.random.normal(0, sigma_red)
    noise_red  = noise_re + 1j * noise_co

    noise_spec = np.zeros(len(sigma), dtype=np.complex128)
    noise_spec[not_zero] = noise_red

    # ---------- back to time domain ----------------------------------
    M = len(noise_spec)            # nper//2 + 1
    N = 2 * (M - 1)                # = nper
    segment = np.fft.irfft(noise_spec, n=N).real
    return segment

# ---------------------------------------------------------------------
# main routine ---------------------------------------------------------
def psdTdiNoiseAveraged(
    tdi_file_path,
    channel   = "X",
    window    = "hann",
    nper      = 4096,
    noverlap  = None,
    average   = "mean",
    scaling   = "density",
    seed      = None
):
    """
    Computes:
        psd_true   – PSD of the original TDI data (WOSA)
        freqs      – frequency vector
        psd_noise  – PSD of a synthetic noise record that has the
                     same length and therefore the same # of Welch
                     averages as psd_true
    """
    # ------------------------------------------------------------------
    # 1) PSD of original TDI time-series
    # ------------------------------------------------------------------
    obs   = TimeSeriesDict.read(tdi_file_path)
    data  = obs[channel].value
    dt    = obs[channel].dt.value
    fs    = 1.0 / dt

    freqs, psd_true = wosa(
        x          = data,
        fs         = fs,
        window     = window,
        nperseg    = nper,
        noverlap   = noverlap,
        scaling    = scaling,
        average    = average,
    )

    n_samples_target = len(data)           # how long our noise must be
    one_seg_psd      = psd_true            # same spectral shape
    noise_segments   = []

    # ------------------------------------------------------------------
    # 2) Stitch together enough segments
    # ------------------------------------------------------------------
    seg_idx = 0
    while sum(len(seg) for seg in noise_segments) < n_samples_target:
        noise_segments.append(
            _noise_segment_from_psd(
                one_seg_psd, fs, nper,
                seed = None if seed is None else seed + seg_idx
            )
        )
        seg_idx += 1

    time_noise = np.concatenate(noise_segments)[:n_samples_target]

    # ------------------------------------------------------------------
    # 3) PSD of the synthetic noise record (same params as true PSD)
    # ------------------------------------------------------------------
    _, psd_noise = wosa(
        x          = time_noise,
        fs         = fs,
        window     = window,
        nperseg    = nper,
        noverlap   = noverlap,
        scaling    = scaling,
        average    = average,
    )

    return psd_true, freqs, psd_noise
