import numpy as np
import math as math
from math import factorial
from scipy.stats import chi2
import warnings
from scipy.special import gammainccinv          # inverse Q(a,x)
from scipy.integrate import quad
from scipy.special import digamma, gamma
from numpy.typing import ArrayLike
from typing import Tuple

def median_pdf(x, N, s):
    """
    PDF of the sample median of N i.i.d. Exp(scale = s) variables.
    """
    x = np.asarray(x, dtype=float)
    y = x / s                       

    # ---------- odd N ----------
    if N % 2 == 1:
        n = (N - 1) // 2
        F = 1 - np.exp(-y)         
        f = np.exp(-y)              
        coeff = factorial(N) / (factorial(n) * factorial(n))
        pdf  = coeff * (F**n) * ((1 - F)**n) * f / s
        return pdf

    # ---------- even N = 2k ----------
    k   = N // 2
    v   = np.exp(y) - 1.0           # upper limit of the inner integral

    # I(k,v) = ∫_0^v t^{k-1}/(1+t) dt
    #        = (-1)^{k-1} [ ln(1+v) + Σ_{j=1}^{k-1} (-1)^j v^j / j ]
    S = np.log1p(v)                 # j = 0 term
    for j in range(1, k):
        S += (-1)**j * v**j / j
    I = (-1)**(k - 1) * S           # the integral value

    pdf = np.exp(-N * y) * I / s    # scale back to Exp(scale = s)

    # numerical guard – tiny negatives can appear from round-off
    pdf = np.where(pdf < 0, 0.0, pdf)
    return pdf   

### The following three functions are used to prove the fact that 
### using Digammma or integration gives same number
### for the median-odd case
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
        num, _ = quad(lambda x: x * y_n(x, n_scalar, s), 0, np.inf, epsrel=1e-8, limit=100)
        den, _ = quad(lambda x:     y_n(x, n_scalar, s), 0, np.inf, epsrel=1e-8, limit=100)
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