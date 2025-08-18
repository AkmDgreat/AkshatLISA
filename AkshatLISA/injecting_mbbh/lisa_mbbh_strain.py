import numpy as np
from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.constants import *

def lisa_mbbh_strain(
    m1=1e6, m2=5e5,
    a1=0.2, a2=0.4, 
    dist=18e3 * PC_SI * 1e6, 
    phi_ref=0.0, 
    f_ref=0.0,
    inc=np.pi/3,
    lam=np.pi/5,
    beta=np.pi/4,
    psi=np.pi/6,
    t_ref=0.5 * YRSID_SI,
    N=17280, dt=5.0,
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
    freq = np.fft.rfftfreq(N, d=dt)

    wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))
    wave_fd = wave_gen(
        m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam,
        beta, psi, t_ref, freqs=freq,
        modes=modes, direct=False, fill=True, squeeze=True, length=1024
    )[0]  # shape (3, len(freq))

    hA = np.fft.irfft(wave_fd[0], n=N)
    hE = np.fft.irfft(wave_fd[1], n=N)
    hT = np.fft.irfft(wave_fd[2], n=N)

    sqrt2 = np.sqrt(2.0)
    sqrt3 = np.sqrt(3.0)
    sqrt6 = np.sqrt(6.0)
    hX = -hA/sqrt2 + hE/sqrt6 + hT/sqrt3
    hY  =      -2*hE/sqrt6 + hT/sqrt3
    hZ =  hA/sqrt2 + hE/sqrt6 + hT/sqrt3

    return t, hX, hY, hZ