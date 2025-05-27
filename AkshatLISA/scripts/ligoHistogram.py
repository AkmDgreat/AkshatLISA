import matplotlib.pyplot as plt

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
