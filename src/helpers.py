import numpy as np
from scipy import stats

import matplotlib.pyplot as plt


#Taken from https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/

def pearsonr_ci(x,y,alpha=0.05, _print=True):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    if _print: print(f"\t corr {r:.3f} in [{lo:.3f}, {hi:.3f}], with p={p:.3f}")
    return lo, hi, p

def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} € [{:.2f},{:.2f}], p = {:.2f} ".format(r, *pearsonr_ci(x,y,_print=False)), xy=(.1, .9), xycoords=ax.transAxes)
