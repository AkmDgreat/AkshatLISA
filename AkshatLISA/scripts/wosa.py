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
        # print(f"median col: {P_stack[: 200]}")

    # elif method == "outlier_rejection":
    #     nseg, nfreq = P_stack.shape
    #     P = np.empty(nfreq, dtype=P_stack.dtype)

    #     # for each frequency bin j, find & reject any segment >3σ (leave-one-out)
    #     for j in range(nfreq):
    #         col = P_stack[:, j]

    #         # compute leave-one-out z-scores
    #         z = np.zeros(nseg, dtype=float)
    #         for i in range(nseg):
    #             # exclude i
    #             others = np.delete(col, i)
    #             μ = others.mean()
    #             std = others.std(ddof=1)
    #             z[i] = 0 if std == 0 else (col[i] - μ) / std
            
    #         # detect outliers
    #         out_idxs = np.where(np.abs(z) > 3)[0]
    #         if out_idxs.size > 0:
    #             mask = np.ones(nseg, dtype=bool) # drop the first flagged outlier
    #             mask[out_idxs[0]] = False 
    #             P[j] = col[mask].mean() # Average the non-outliers
    #         else:
    #             P[j] = col.mean() # no outlier → simple mean
            
    #         if j==200: 
    #             print(f"col: {col}")
    #             print(f"z: {z}")
    #             print(f"out_idxs: {out_idxs}")
    #             print(f"P[j]: {P[j]}")
    elif method == "outlier_rejection":
        k_passes = 2                   # run the test-and-drop loop twice
        nseg, nfreq = P_stack.shape
        P = np.empty(nfreq, dtype=P_stack.dtype)

        for j in range(nfreq):
            keep = np.ones(nseg, dtype=bool)  # start with every segment kept

            # run up to k_passes; each pass can delete at most ONE segment
            for _ in range(k_passes):
                col = P_stack[keep, j]        # current survivors
                if col.size < 3:              # need ≥3 points for σ with ddof=1
                    break
                μ   = col.mean()
                σ   = col.std(ddof=1)
                if σ == 0:                    # identical values → nothing to reject
                    break

                z   = np.abs((col - μ) / σ)   # standard z-scores (not leave-one-out)
                out = np.where(z > 3)[0]
                if out.size == 0:             # no outlier → stop early
                    break

                # drop *one* outlier (first or worst – here we pick the worst)
                worst_local  = z.argmax()     # index in `col`
                worst_global = np.flatnonzero(keep)[worst_local]
                keep[worst_global] = False    # mark as rejected and loop again

            P[j] = P_stack[keep, j].mean()    # final average of survivors

    else:
        raise ValueError("method must be 'mean' or 'median' or 'outlier_rejection'")
    
    f = np.fft.rfftfreq(nfft, d=1.0/fs)
    # print("Exiting custom wosa")

    # return the "how many segments were averaged" array
    # it is constant for WOSA, but for logPSD, it is different 
    nseg = np.full(nfft//2 + 1, nseg)

    # fact: f.length = P.length = nseg.length
    return f, P, P_stack, nseg