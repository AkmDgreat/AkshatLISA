import numpy as np
import scipy.stats as st
import scipy.optimize as opt

def mle_df(noise_vals, mu=None, round_even=True):
    """
    Maximum-likelihood estimate of χ² degrees-of-freedom (and μ if desired).

    Parameters
    ----------
    noise_vals : 1-D array
        Realisations of the PSD estimator at one frequency bin.
    mu : float or None, optional
        If None (default) μ is treated as unknown and estimated jointly.
        If a positive number is supplied, that value is held fixed.
    round_even : bool, default True
        Whether to round df̂ to the nearest even integer (2 K) as in the
        ideal independent-segment Welch model.

    Returns
    -------
    df_hat : float or int
        Estimated degrees-of-freedom (even integer if round_even=True).
    mu_hat : float
        Estimated (or supplied) μ.
    """

    noise_vals = np.asarray(noise_vals)
    if np.any(noise_vals <= 0):
        raise ValueError("noise_vals must be positive for log-pdf evaluation")

    # ---------- log-likelihood -----------------------------------------
    if mu is None:                                   # μ unknown  → 2-parameter fit
        def nll(params):
            df, mu_ = params
            if df <= 0 or mu_ <= 0:
                return np.inf
            scale = 2 * mu_ / df
            return -st.gamma.logpdf(noise_vals,
                                     a=df/2,
                                     loc=0,
                                     scale=scale).sum()

        # method-of-moments starting guesses
        m, s2 = noise_vals.mean(), noise_vals.var(ddof=1)
        df0   = 2 * m**2 / s2
        res = opt.minimize(nll,
                           x0=[df0, m],
                           bounds=[(1e-3, None), (1e-12, None)])
        df_hat, mu_hat = res.x

    else:                                            # μ fixed  → 1-parameter fit
        if mu <= 0:
            raise ValueError("mu must be positive")
        def nll(df):
            if df <= 0:
                return np.inf
            scale = 2 * mu / df
            return -st.gamma.logpdf(noise_vals,
                                     a=df/2,
                                     loc=0,
                                     scale=scale).sum()

        # scalar optimisation is enough
        m, s2 = noise_vals.mean(), noise_vals.var(ddof=1)
        df0   = 2 * m**2 / s2                        # reasonable starting point
        ub    = df0 * 50                 # finite upper bound
        res = opt.minimize_scalar(nll,
                                  bracket=(df0/5, df0*5),
                                  bounds=(1e-3, ub),
                                  method='bounded')
        df_hat, mu_hat = res.x, mu

    # ---------- round if desired ---------------------------------------
    if round_even:
        df_hat = int(round(df_hat / 2.0)) * 2

    return df_hat, mu_hat
