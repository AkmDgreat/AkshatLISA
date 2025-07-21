import numpy as np
from bbhx.waveformbuild import BBHWaveformFD

def lisa_mbbh_strain(
    m1, m2, a1, a2, dist, phi_ref, f_ref,
    inc, lam, beta, psi, t_ref,
    N=10000, dt=5.0,
    modes=[(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
):
    """
    Generate time-domain LISA TDI channels (A, E, T) from BBH parameters.

    Returns:
        t : np.ndarray, time array (seconds)
        hA, hE, hT : np.ndarray, real strain time-series
    """
    fs = 1 / dt
    t = np.arange(N) / fs
    freq_new = np.fft.rfftfreq(N, d=dt)

    wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))
    wave_fd = wave_gen(
        m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam,
        beta, psi, t_ref, freqs=freq_new,
        modes=modes, direct=False, fill=True, squeeze=True, length=1024
    )[0]  # shape (3, len(freq_new))

    hA = np.fft.irfft(wave_fd[0], n=N)
    hE = np.fft.irfft(wave_fd[1], n=N)
    hT = np.fft.irfft(wave_fd[2], n=N)

    return t, hA, hE, hT