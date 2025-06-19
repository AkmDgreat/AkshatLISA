import numpy as np
from math import factorial
from scipy.stats import chi2

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
    # f = lambda x: np.exp(-x/scale)
    F = lambda x: 1 - np.exp(-x/scale)

    return median_pdf(v, N, f, F)

# This works for both histogram and any function
def confidence_interval(data, confidence=0.90):
    """
    Compute the two-sided confidence interval for a 1D array.

    Parameters
    ----------
    data : array-like
        Your sample of values.
    confidence : float, optional
        Desired confidence level between 0 and 1 (default 0.90).

    Returns
    -------
    lower : float
        The lower confidence limit (e.g. 5th percentile for C=0.90).
    upper : float
        The upper confidence limit (e.g. 95th percentile for C=0.90).
    """
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError("`data` must contain at least one value")
    if not (0 < confidence < 1):
        raise ValueError("`confidence` must be between 0 and 1")

    alpha = 1.0 - confidence
    lower_q = 100 * alpha / 2
    upper_q = 100 * (1 - alpha / 2)

    lower = np.percentile(data, lower_q)
    upper = np.percentile(data, upper_q)
    return lower, upper

from scipy.special import gammainccinv          # inverse Q(a,x)
from numpy.typing import ArrayLike
from typing import Tuple

def psd_cl(
    S_uu: ArrayLike,          # estimated PSD values
    W:    ArrayLike,          # number of averages per bin
    c:    float = 0.10        # tail probability →  (1-c)·100 % band
) -> Tuple[np.ndarray, np.ndarray]:

    if S_uu.shape != W.shape:
        raise ValueError("S_uu and W must have the same shape")

    m = W - 1.0
       
    # upper = (1.0 + c) / 2.0    
    upper = c / 2.0
    gamma_plus  = gammainccinv(m, upper)  
    S_minus = W * S_uu /  gamma_plus

    # lower = (1.0 - c) / 2.0  
    lower = 1-c/2.0
    gamma_minus = gammainccinv(m, lower)  
    S_plus  = W * S_uu / gamma_minus  

    return S_minus, S_plus

def psd_ci(psd_hat, W, c=0.10):
    """
    (1-c) confidence interval for a Welch / WOSA PSD estimate.

    Parameters
    ----------
    psd_hat : array_like
        Point estimate \hat{S}_{uu}(k)
    W : int or array_like
        Equivalent number of averages W_k (nu = 2W DoF)
    c : float, optional
        Tail probability; c=0.10 -> 90 % band

    Returns
    -------
    S_lower, S_upper : ndarray
        Lower and upper confidence curves (same shape as psd_hat)
    """
    psd_hat = np.asarray(psd_hat, dtype=float)
    W = np.asarray(W, dtype=float)

    chi2_upper = chi2.ppf(c/2.0,     2*W)   # 5th percentile for c=0.10
    chi2_lower = chi2.ppf(1-c/2.0,  2*W)   # 95th percentile

    S_lower = 2*W * psd_hat / chi2_lower     # < psd_hat
    S_upper = 2*W * psd_hat / chi2_upper     # > psd_hat
    return S_lower, S_upper