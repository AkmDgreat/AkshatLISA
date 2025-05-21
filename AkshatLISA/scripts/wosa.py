import numpy as np

def wosa(x,
             fs: float = 1.0,
             nperseg: int = 256,
             noverlap: int | None = None,
             window: str | np.ndarray = "hann",
             detrend: str = "constant",
             scaling: str = "density"
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
    Pxx : ndarray
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
    if nseg <= 0:
        raise ValueError("Segment configuration yields no segments.")

    # --- window ---------------------------------------------------------------
    if isinstance(window, str):
        if window.lower() == "hann":
            win = np.hanning(nperseg)
        elif window.lower() == "hamming":
            win = np.hamming(nperseg)
        else:
            raise ValueError(f"Unknown window '{window}'.")
    else:
        win = np.asarray(window, dtype=float)
        if win.shape != (nperseg,):
            raise ValueError("Window length must equal nperseg.")

    U = (win**2).sum()                      # window power for normalization
    scale = 1.0 / (fs * U) if scaling == "density" else 1.0 / U

    # --- allocate output accumulator -----------------------------------------
    nfft = nperseg
    P_acc = np.zeros(nfft//2 + 1, dtype=float)

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
        P = np.abs(Xf)**2
        print(P)
        P_acc += P

    Pxx = (P_acc / nseg) * scale
    f = np.fft.rfftfreq(nfft, d=1.0/fs)
    print("Exiting custom wosa")
    return f, Pxx