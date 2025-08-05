import numpy as np
from scipy.stats import chi2
from scipy.integrate import simpson

def pdf(x, N, s, method):
    """
    PDF for PSD estimate with "N" segments, and true PSD "s"
    """
    if method == 'mean':
        return mean_pdf(x, N, s)
    elif method == 'median':
        return median_pdf(x, N, s)

def mean_pdf(x, N, s): 
    """
    PDF for mean-based methods with "N" segments, and true PSD "s"
    """

    x = np.asarray(x, dtype=float)
    df = 2 * N         
    pdf = (df / s) * chi2.pdf((df/s) * x, df)
    return pdf

def median_pdf(x, N, s):
    """
    PDF for median-based methods with "N" segments, and true PSD is "s"  
    """
    x = np.asarray(x, dtype=float)
    y = x / s                       

    if N % 2 == 1:
        n = (N - 1) // 2
        F = 1 - np.exp(-y)         
        f = np.exp(-y)              
        pdf = (F**n) * ((1 - F)**n) * f / s
        
    else: 
        k   = N // 2
        v   = np.exp(y) - 1.0           

        # I(k,v) = ∫_0^v t^{k-1}/(1+t) dt
        #        = (-1)^{k-1} [ ln(1+v) + Σ_{j=1}^{k-1} (-1)^j v^j / j ]
        S = np.log1p(v)                 # j = 0 term
        for j in range(1, k):
            S += (-1)**j * v**j / j
        I = (-1)**(k - 1) * S          

        pdf = np.exp(-N * y) * I / s    # scale back to Exp(scale = s)

        # numerical guard – tiny negatives can appear from round-off
        pdf = np.where(pdf < 0, 0.0, pdf)

    pdf /= simpson(pdf, x=x) # normalise
    return pdf