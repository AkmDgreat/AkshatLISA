import numpy as np
import scipy.signal

def wosa(x,
             fs: float = 1.0,
             nperseg: int = 256,
             noverlap: int | None = None,
             window: str | np.ndarray = "hann",
             detrend: str = "constant",
             scaling: str = "density",
             method: str = "mean"
):
    """
    Estimate the one-sided Power Spectral Density (PSD) of a real-valued
    time-series using Welch / WOSA.

    Parameters
    ----------
    x : 1-D array_like
        The data sequence (time-domain samples).
    fs : float, default 1.0
        Sampling frequency in Hz.
    nperseg : int, default 256
        Number of samples per segment.
    noverlap : int, optional
        Number of points to overlap successive segments (defaults to nperseg//2).
    window : str | array_like, default "hann"
        Window applied to each segment (string passed to `np.hanning`, etc.,
        or a NumPy array of length *nperseg*).
    detrend : {"constant", "linear"} or callable, default "constant"
        Detrending method applied to each segment.
    scaling : {"density", "spectrum"}, default "density"
        ``"density"`` returns PSD [power/Hz]; ``"spectrum"`` returns power.

    Returns
    -------
    f : (nfft//2 + 1,) ndarray
        Array of positive sample frequencies.
    P : ndarray
        PSD (or power spectrum) estimated at `f`.
    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input must be 1-D.")

    if nperseg > len(x):
        raise ValueError("nperseg may not exceed input length.")

    # --- segmentation & overlap ----------------------------------------------
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    nseg = 1 + (len(x) - nperseg) // step  # integer division
    # print(f"Number of segments: {nseg}")
    if nseg <= 0:
        raise ValueError("Segment configuration yields no segments.")

    # --- window ---------------------------------------------------------------
    if isinstance(window, (str, tuple)):
        try:
            # this will handle 'hann', ('kaiser', β), 'nuttall', ('nuttall', False), etc.
            win = scipy.signal.get_window(window, nperseg)
        except ValueError:
            raise ValueError(f"Unknown window spec {window!r}")
    else:
        win = np.asarray(window, dtype=float)
        if win.shape != (nperseg,):
            raise ValueError(f"Window length must equal nperseg ({nperseg}), got {win.shape}")

    U = (win**2).sum()                      # window power for normalization
    scale = 1.0 / (fs * U) if scaling == "density" else 1.0 / U

    # --- allocate output accumulator -----------------------------------------
    nfft = nperseg
    P_stack = np.empty((nseg, nfft//2 + 1), dtype=float)

    # --- iterate over segments -----------------------------------------------
    for k in range(nseg):
        start = k * step
        segment = x[start:start + nperseg].copy()

        # detrend
        if detrend == "constant":
            segment -= segment.mean()
        elif detrend == "linear":
            t = np.arange(nperseg)
            segment -= np.polyval(np.polyfit(t, segment, 1), t)
        elif callable(detrend):
            segment = detrend(segment)
        elif detrend is not None:
            raise ValueError("detrend must be 'constant', 'linear', callable, or None.")

        segment *= win

        # FFT and (one-sided) periodogram
        Xf = np.fft.rfft(segment, n=nfft)
        P_stack[k] = np.abs(Xf)**2

    # --- fold negative-frequency power & scale each segment -------------
    P_stack[:, 1:-1] *= 2       # one-sided correction (skip DC & Nyquist)
    P_stack *= scale            # convert |X|² → PSD [power/Hz] (or power)

    # mean or median across segments
    if method == "mean":
        P = P_stack.mean(axis=0)
    elif method == "median":
        P = np.median(P_stack, axis=0)

    elif method == "outlier_rejection":
        # 1. mean & (sample) std-dev per frequency bin
        mu  = P_stack.mean(axis=0)                 # shape (n_freq,)
        sig = P_stack.std(axis=0, ddof=1)

        # 2. boolean mask of inliers  |x-μ| ≤ 3σ
        z = np.abs(P_stack - mu) / sig             # broadcasts over segments axis
        mask = z <= 3                              # shape (n_seg, n_freq)

        # 3. average only the inliers; fall back to μ if all points were rejected
        inlier_count = mask.sum(axis=0)            # how many survived in each bin
        # avoid divide-by-zero: where inlier_count==0, keep the original mean
        P = np.where(inlier_count,
                    (P_stack * mask).sum(axis=0) / inlier_count,
                    mu)
    else:
        raise ValueError("method must be 'mean' or 'median' or 'outlier_rejection'")
    
    f = np.fft.rfftfreq(nfft, d=1.0/fs)
    # print("Exiting custom wosa")

    # return the "how many segments were averaged" array
    # it is constant for WOSA, but for logPSD, it is different 
    nseg = np.full(nfft//2 + 1, nseg)

    # fact: f.length = P.length = nseg.length
    return f, P, P_stack, nseg