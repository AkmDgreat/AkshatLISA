import numpy as np
from scipy.signal import welch            # swap this for your own `wosa` if you like
from scripts.wosa import wosa

def logpsd(x,
           fs: float,
           M: int      = 4,
           alpha: int  = 2,
           L1: int | None = None,
           window: str = "hann",
           overlap: float = 0, 
           scaling: str = "density"):
    """
    Minimum-error, low-correlation *logPSD* estimator.

    Parameters
    ----------
    x        : 1-D ndarray
        Real-valued time series.
    fs       : float
        Sampling frequency [Hz].
    M        : int, default 4
        Number of window-affected bins to discard (see eq. 2.16).
    alpha    : int, default 2
        Spectral-resolution parameter (sets Q = M/α, eqs. 2.17–2.18).
    L1       : int or None
        Length of the first segment L₁.  If None, the code chooses the
        largest power-of-two ≤ len(x)//8.
    window   : str, default "hann"
        scipy / NumPy window name passed to Welch.
    overlap  : float in [0, 1), default 0
        Fractional overlap between successive segments.
    scaling  : {"density", "spectrum"}, default "density"
        Welch scaling.
    
    Returns
    -------
    f_opt    : ndarray
        Optimised (log-spaced) frequency grid.
    P_opt    : ndarray
        logPSD estimate at those frequencies.
    """

    N   = len(x)
    dt  = 1.0 / fs
    # ------------------------------------------------------------------
    # 1. Choose the first-segment length  L₁  (user-supplied or heuristics)
    # ------------------------------------------------------------------
    if L1 is None:
        # a simple—but sensible—default: 1/8th of the record, power-of-two
        L1 = 1 << (N.bit_length() - 4)      # divide by 16 ≈ 2⁴
    if L1 < M + 1:
        raise ValueError("Chosen L1 is too small: must be > M.")
    
    f0 = M / (L1 * dt)                      # eq. 2.16
    Q  = int(M / alpha)                    # ≤ M/α   (eqs. 2.17–2.18)
    r  = (2*Q - 1) / (2*Q + 1)             # geometric ratio in eq. 2.18
    
    # ------------------------------------------------------------------
    # 2. Build the optimised grid {L_k}, {f_k}
    # ------------------------------------------------------------------
    Lk_list, fk_list = [], []
    string_arr = []
    k = 1          
    while True:
        # segment length L_k
        if k == 1 or k <= Q:
            Lk = L1
            string_arr.append("L_1")
        else:
            Lk = int(np.floor(r**(k-Q) * L1))   # eq. 2.18
            string_arr.append(f"({r})^{k-Q} L_1")
            if Lk < M + 1:                      # segments got too short
                break
        
        fk = M / (Lk * dt)                      # associated freq (eq. 2.16)
        if fk >= fs/2:                          # stop at Nyquist
            break
        
        Lk_list.append(Lk)
        fk_list.append(fk)
        k += 1
    
    f_opt = np.asarray(fk_list)
    P_opt = np.empty_like(f_opt)
    
    print(string_arr)
    print(f"Lk_list: {Lk_list}")
    print(f"fk_list: {fk_list}")
    # ------------------------------------------------------------------
    # 3. Estimate the PSD at every (L_k, f_k)
    #    • run Welch/WOSA with segment length L_k
    #    • discard the first M bins           (paper step 5)
    #    • pick the bin closest to f_k
    # ------------------------------------------------------------------
    nseg_arr = np.empty_like(f_opt)
    for j, (Lk, fk) in enumerate(zip(Lk_list, f_opt)):
        noverlap = int(overlap * Lk)
        f, Pxx, _, nseg = wosa(x,
                       fs=fs,
                       window=window,
                       nperseg=Lk,
                       noverlap=noverlap,
                       scaling=scaling)
        f, Pxx = f[M:], Pxx[M:]                # drop window-affected bins
        idx    = np.abs(f - fk).argmin()       # nearest bin
        P_opt[j] = Pxx[idx]
        nseg_arr[j] = nseg[0] # cuz each of the nseg returned from wosa is an array 
    
    return f_opt, P_opt, nseg_arr