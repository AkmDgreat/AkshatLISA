import numpy as np
from math import factorial
from scipy.stats import chi2
import warnings
from scipy.special import gammainccinv          # inverse Q(a,x)
from scipy.integrate import quad
from scipy.special import digamma, gamma
from numpy.typing import ArrayLike
from typing import Tuple

# def median_pdf(v, N, f, F):
#     """
#     Density of the sample median for an odd sample size N.

#     Parameters
#     ----------
#     v : float or array_like
#         Point(s) where the pdf is evaluated.
#     N : int  (must be odd)
#         Sample size used to form the median.
#     f : callable
#         Population probability-density function  f(x).
#     F : callable
#         Population cumulative-distribution function  F(x).

#     Returns
#     -------
#     pdf : float or ndarray
#         Value(s) of the density at v.
#     """
#     if N % 2 == 0:
#         warnings.warn(f"N={N} is even, not recommended", UserWarning)
#         n = N / 2.0
#     else:
#         n = (N - 1) // 2                 # half-size parameter
    
#     print(f"n: {n}")
#     v = np.asarray(v)
#     coeff = factorial(N) / (factorial(n) * factorial(n))   # (N)! / (n! n!)
#     return coeff * (F(v)**n) * ((1 - F(v))**n) * f(v)

def harmonic_factor(N, method='lower'):
    if N % 2:                       # odd: only one definition
        n = (N - 1) // 2
        return digamma(N + 1) - digamma(n + 1)

    # Several methods to choose from in the even case
    if method == 'scott':
        m = N / 2.0
        print(f"m is {m}") 
        return digamma(N + 1) - digamma(m)

    m = N // 2                      # even N = 2m
    if method == 'lower':
        return digamma(N + 1) - digamma(m + 1)
    elif method == 'upper':
        return digamma(N + 1) - digamma(m)
    elif method == 'average':
        lower  = digamma(N + 1) - digamma(m + 1)
        upper  = digamma(N + 1) - digamma(m)
        return 0.5*(lower + upper)
    else:
        raise ValueError("method must be 'lower', 'upper', or 'average'")

def order_stat_pdf(v, N, k, f, F):
    """
    PDF of the k-th order statistic X_(k) from a sample of size N.

    Parameters
    ----------
    v : float or array_like
    N : int             Sample size (N >= 1) (or float)
    k : int             Order (1 <= k <= N)
    f : callable        Population pdf  f(x)
    F : callable        Population cdf  F(x)
    """
    if not (1 <= k <= N):
        raise ValueError("k must be between 1 and N inclusive")
    v = np.asarray(v)

    if isinstance(k, float):
        # coeff = gamma(N) / (gamma(k - 1) * gamma(N - k))
        coeff = gamma(N) / (gamma(k - 1) * gamma(k-1))
        # ealier: I was using //
    else:
        coeff = factorial(N) / (factorial(k - 1) * factorial(N - k))

    # return coeff * (F(v) ** (k - 1)) * ((1 - F(v)) ** (N - k)) * f(v)
    return coeff * (F(v) ** (k - 1)) * ((1 - F(v)) ** (k - 1)) * f(v)

def median_pdf(v, N, f, F, method='lower'):
    """PDF of the sample median for any N, using the chosen method."""
    if N % 2:               # odd sample size
        k = (N + 1) / 2
        return order_stat_pdf(v, N, k, f, F)
    else:                   # even
        if method == 'scott':
            n = (N+1) / 2
            print(f"n = {n}")
            return order_stat_pdf(v, N, n, f, F)
        n = N // 2
        if method == 'lower':
            return order_stat_pdf(v, N, n, f, F)
        elif method == 'upper':
            return order_stat_pdf(v, N, n + 1, f, F)
        elif method == 'average':
            # average handled by avg_median_exp_pdf, should never get here
            raise RuntimeError("Should have been caught in median_exp_pdf")
        else:
            raise ValueError("method must be 'lower', 'upper', or 'average'")

def _pdf_average_median_exp(v, N, scale):
    """
    PDF of the average of X_(n) and X_(n+1) for an Exp(scale) parent,
    where N = 2n is even.
    """
    print("inside new function")
    assert N % 2 == 0
    n        = N // 2
    coeff    = factorial(N) / (factorial(n - 1) ** 2) * 2.0
    v        = np.asarray(v, dtype=float)
    pdf_vals = np.zeros_like(v)

    inv_scale = 1.0 / scale

    for i, m in enumerate(v):
        if m <= 0.0:                  # support is m > 0
            continue

        def integrand(u):
            mp = m + u                # m+u ≥ m ≥ 0
            mm = m - u                # 0 ≤ mm ≤ m
            # Exploit exponential formulas:  F(x)=1–e^{–x/λ}, 1–F(x)=e^{–x/λ}, f(x)=e^{–x/λ}/λ
            exp_mp = np.exp(-mp * inv_scale)
            exp_mm = np.exp(-mm * inv_scale)
            return ((1.0 - exp_mp) ** (n - 1)) * (exp_mm ** (n - 1)) * \
                   (exp_mp * exp_mm) * (inv_scale ** 2)

        # integral only up to m (beyond that mm<0 ⇒ integrand ≡ 0)
        res, _ = quad(integrand, 0.0, m, epsabs=1e-9, epsrel=1e-9)
        pdf_vals[i] = coeff * res

    return pdf_vals

def avg_median_exp_pdf(v, N, scale=1.0, tol=1e-9):
    """
    Numerically stable pdf of M = (X_(m)+X_(m+1))/2,  N = 2m,  X~Exp(scale).
    """
    if N % 2:
        raise ValueError("N must be even for the average-median definition")

    n  = N // 2                # m in the notation above
    v  = np.atleast_1d(v).astype(float)
    pdf = np.zeros_like(v)

    coeff = factorial(N) / (factorial(n - 1) ** 2) * 2.0
    lam   = scale

    for i, m in enumerate(v):
        if m <= 0.0:
            pdf[i] = 0.0
            continue

        # integrate u from 0 .. m  (beyond m integrand=0)
        def log_integrand(u):
            mp = m + u
            mm = m - u

            # log-pdf of Exp(scale)
            log_f_mp = -mp / lam - np.log(lam)
            log_f_mm = -mm / lam - np.log(lam)

            # log-cdf and log(1-cdf) (use log1p for accuracy)
            log_F_mp     = np.log1p(-np.exp(-mp / lam))
            log_one_F_mm = np.log1p(-np.exp(-mm / lam))

            return ((n - 1) * log_F_mp
                    + (n - 1) * log_one_F_mm
                    + log_f_mp + log_f_mm)

        def integrand(u):
            return np.exp(log_integrand(u))

        pdf[i], _ = quad(integrand, 0.0, m, epsabs=tol)

    return coeff * pdf if pdf.ndim else pdf.item()

def median_exp_pdf(v, N, scale=1.0, method='lower'):
    """
    Wrapper that calls:
      • order_stat_pdf   for lower / upper (and for odd N),
      • avg_median_exp_pdf for 'average' when N is even.
    """
    # parent pdf/cdf
    f = lambda x: np.exp(-x / scale) / scale
    F = lambda x: 1 - np.exp(-x / scale)

    if method == 'average' and (N % 2 == 0):
        # return avg_median_exp_pdf(v, N, scale)
        return _pdf_average_median_exp(v, N, scale) # new code
    else:
        return median_pdf(v, N, f, F, method)

def y_n(x, n, s):
    f = np.exp(-x/s) / s  # PDF
    F = 1 - np.exp(-x/s)  # CDF
    return (F**n) * (1-F)**n * f      

def ratio_numerical(n, s=1.0):
    """
    n : int or array-like of ints  (n ≥ 1)
    s : scale (default 1)
    returns : float or ndarray   —  (∫ x y_n / ∫ y_n)
    """
    n_arr = np.asarray(n, dtype=float)                # works for scalar or array
    out   = np.empty_like(n_arr, dtype=float)

    # iterate over every element (ndim-safe)
    for idx, n_scalar in np.ndenumerate(n_arr):
        num, _ = quad(lambda x: x * y_n(x, n_scalar, s), 0, np.inf)
        den, _ = quad(lambda x:     y_n(x, n_scalar, s), 0, np.inf)
        out[idx] = num / den

    # return a scalar if a scalar went in
    return out.item() if np.isscalar(n) else out

def ratio(n, s=1.0):
    """
    n : int or 1-D array of ints (n ≥ 1)
    s : positive scale (default 1)
    returns s · [ψ(2n+2) - ψ(n+1)]
    """
    n = np.asarray(n, dtype=float)
    return s * (digamma(2*n + 2) - digamma(n + 1))

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