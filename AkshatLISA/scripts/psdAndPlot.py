import matplotlib.pyplot as plt
import scipy.signal as sig
from scripts.lpsd import lpsd

# plot the figures horizontally
def psd_and_plot_hor(data, t, dt, nper, title, window='hann', scaling='density'):
    # 1×4 subplots, make it wide enough
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # prep common quantities
    fs    = 1.0 / dt
    f_min = 1.0 / (nper * dt)
    f_max = fs / 2.0

    # 1) time-domain
    ax = axes[0]
    ax.plot(t, data)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(title)
    ax.set_title("Time series")

    # 2) WOSA PSD (log–log)
    f_w, psd_w = sig.welch(
        data,
        fs=fs,
        window=window,
        nperseg=nper,
        noverlap=nper//2,
        scaling=scaling
    )
    ax = axes[1]
    ax.loglog(f_w, psd_w)
    ax.set_xlim(f_min, f_max)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    ax.set_title("WOSA (log–log)")

    # 3) LPSD
    f_l, psd_l = lpsd(
        data,
        fs=fs,
        window=window,
        scaling=scaling
    )
    ax = axes[2]
    ax.loglog(f_l, psd_l)
    ax.set_xlim(f_min, f_max)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    ax.set_title("LPSD")

    # 4) WOSA PSD (linear)
    ax = axes[3]
    ax.plot(f_w, psd_w)
    ax.set_xlim(f_min, f_max)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    ax.set_title("WOSA (linear)")

    fig.tight_layout()

# plot the figures vertically 
def psd_and_plot(data, t, dt, nper, title, window='hann', scaling='density'):
    # 1) time-domain
    plt.figure()
    plt.plot(t, data)
    plt.xlim(t[0], t[t.size - 1])
    plt.xlabel("Time [s]")
    plt.ylabel(title)

    # prepare freq-axis limits
    fs = 1.0 / dt
    f_min = 1.0 / (nper * dt)
    f_max = fs / 2.0

    f_w, psd_w = sig.welch(
        data,
        fs=fs,
        window=window,
        nperseg=nper,
        noverlap=nper//2,
        scaling=scaling
    )

    # 2) WOSA psd log scale
    plt.figure()
    plt.loglog(f_w, psd_w)
    plt.xlabel("Frequency [Hz]")
    plt.xlim(f_min, f_max)
    plt.ylabel("PSD [1/Hz]")
    plt.title("WOSA, log scale")

    # 3) LPSD PSD
    f_l, psd_l = lpsd(
        data,
        fs=fs,
        window=window,
        scaling=scaling
    )
    plt.figure()
    plt.loglog(f_l, psd_l)
    plt.xlabel("Frequency [Hz]")
    plt.xlim(f_min, f_max)
    plt.ylabel(f"PSD [1/Hz] (LPSD)")
    plt.ylabel("PSD [1/Hz]")
    plt.title("LPSD")

    # 4) WOSA PSD linear scale
    plt.figure()
    plt.plot(f_w, psd_w)
    plt.xlabel("Frequency [Hz]")
    plt.xlim(f_min, f_max)
    plt.ylabel("PSD [1/Hz]")
    plt.title("WOSA, linear scale")

    plt.tight_layout()