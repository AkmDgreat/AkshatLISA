import sys
import numpy as np
import matplotlib.pyplot as plt

# t_max = 365 days
# using default args: we will choose 4 time stamps (either Poisson or equally spaced) 
# between t = 0 and t = 365 days
# and these times will be our glitch times 
def glitch_times(glitch_rate=4, t0=0.0, t_max=31536000, glitch_type='Poisson', equal_space=20000):  # TODO figure out these defaults times
    """return glitch injection times from t0 to t_max for the given distribution type"""

    n_glitches = glitch_rate

    if glitch_type == 'Poisson' or glitch_type == 'poisson':
        t_max_day = int(t_max / (24 * 3600))
        all_events = {}
        for i in range(t_max_day):
            n_per_day = np.random.poisson(n_glitches, 1)
            interval_n = np.random.exponential(1 / n_glitches, n_per_day)
            all_events[i] = {'n glitches': n_per_day, 'time interval btwn events [s]': interval_n * 24 * 3600}

        t, glitch_times_list = t0, []
        for k in all_events.keys():
            n_events, time_sep = all_events[k]['n glitches'], all_events[k]['time interval btwn events [s]']
            for t_sep in time_sep:
                t = t + t_sep
                glitch_times_list.append(t)

    elif glitch_type == 'Equal Spacing':
        n, glitch_times_list = 1, []
        for i in range(t0, t_max):

            if i == n*equal_space:
                glitch_times_list.append(i)
                n += 1

    else:
        sys.exit(f"Not an available distribution {glitch_type}")

    return np.array(glitch_times_list)


def amplitude_dist(avg_amp=10**-5, std_amp=10**-6, n_samples=10, type_dist='Gaussian', amp_set=[10**-10, 10**-5]):
    """return n_samples of amplitude, taken from a Gaussian distribution"""
    print(amp_set)
    if type_dist == 'Gaussian' or type_dist == 'gaussian':
        amp = np.random.normal(avg_amp, std_amp, n_samples)
        sign = np.random.random(size=n_samples) < 0.5  # Random choice of the glitch amplitude sign.
        amp[sign] *= -1
    elif type_dist == 'Set' or type_dist == 'set':
        try:
            amp = np.linspace(amp_set[0], amp_set[1], n_samples)
        except TypeError or IndexError:
            amp = np.linspace(amp_set, n_samples)
        sign = np.random.random(size=n_samples) < 0.5  # Random choice of the glitch amplitude sign.
        amp[sign] *= -1
    else:
        sys.exit(f"Not an available amplitude distribution {type_dist}")

    return amp

# sets the amplitude distribution 
# The time distribution sets the timing of the glitches
# the amp distribution sets the amplitude of the glitches 


def betas_dist(scale=4, n_samples=10, type_dist='Exponential', beta_set=[0.001, 100]):
    """return n_samples of beta (time dampening factor for the glitches)"""

    if type_dist == 'Exponential' or type_dist == 'exponential':
        betas = np.random.exponential(scale, n_samples)
    elif type_dist == 'Set' or type_dist == 'set':
        try:
            betas = np.linspace(beta_set[0], beta_set[1], n_samples)
        except TypeError or IndexError:
            betas = np.linspace(beta_set, n_samples)

    return betas


