import numpy as np
from math import factorial

def median_pdf(v, N, f, F):
    """
    Density of the sample median for an odd sample size N.

    Parameters
    ----------
    v : float or array_like
        Point(s) where the pdf is evaluated.
    N : int  (must be odd)
        Sample size used to form the median.
    f : callable
        Population probability-density function  f(x).
    F : callable
        Population cumulative-distribution function  F(x).

    Returns
    -------
    pdf : float or ndarray
        Value(s) of the density at v.
    """
    if N % 2 == 0:
        raise ValueError(f"Sample size N={N} must be odd so the median is unique")

    n = (N - 1) // 2                 # half-size parameter
    v = np.asarray(v)

    coeff = factorial(N) / (factorial(n) * factorial(n))   # (2n+1)! / (n! n!)
    return coeff * (F(v)**n) * ((1 - F(v))**n) * f(v)

def median_exp_pdf(v, N, scale=1.0):
    """
    PDF of the sample median when the parent distribution is Exp(scale).

    Parameters
    ----------
    v     : array_like
        Points at which to evaluate the density.
    N     : int (must be odd)
        Sample size whose median you are modelling.
    scale : float, optional
        Mean (scale) of the exponential distribution.  Default 1.0.

    Returns
    -------
    pdf : ndarray
        Density of the sample median evaluated at v.
    """
    # parent pdf and cdf for Exp(scale)
    f = lambda x: np.exp(-x/scale) / scale
    F = lambda x: 1 - np.exp(-x/scale)

    return median_pdf(v, N, f, F)