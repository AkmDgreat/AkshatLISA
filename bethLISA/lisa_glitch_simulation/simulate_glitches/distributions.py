import numpy as np


def glitch_times_poisson(glitch_rate=4, t0=0.0, t_max=31536000, seed=None):
    n_glitches = glitch_rate

    t_max_day = int(t_max / (24 * 3600))
    all_events = {}

    np.random.seed(seed)

    for i in range(t_max_day):
        n_per_day = np.random.poisson(n_glitches, 1)
        interval_n = np.random.exponential(1 / n_glitches, n_per_day)
        all_events[i] = {
            "n glitches": n_per_day,
            "time interval btwn events [s]": interval_n * 24 * 3600,
        }

    t, glitch_times_list = t0, []

    for k in all_events.keys():
        time_sep = (all_events[k]["time interval btwn events [s]"],)
        for t_sep in time_sep:
            t += t_sep
            glitch_times_list.append(t)

    return np.array(glitch_times_list)


def glitch_times_equal_spacing(equal_space=20000, t0=0.0, t_max=31536000):
    i, glitch_times_list = t0, []

    while i + equal_space <= t_max:
        i += equal_space
        glitch_times_list.append(i)

    return np.array(glitch_times_list)


def amplitude_dist_set(n_samples=10, amp_set=[10**-10, 10**-5], seed=None):
    np.random.seed(seed)

    try:
        amp = np.linspace(amp_set[0], amp_set[1], n_samples)
    except TypeError or IndexError:
        # TODO: Maybe add sys.ext?
        amp = np.linspace(amp_set, n_samples)
    sign = (
        np.random.random(size=n_samples) < 0.5
    )  # Random choice of the glitch amplitude sign.
    amp[sign] *= -1

    return amp


def amplitude_dist_gaussian(
    n_samples=10, avg_amp=10**-5, std_amp=10**-6, seed=None
):
    np.random.seed(seed)

    amp = np.random.normal(float(avg_amp), float(std_amp), n_samples)

    sign = (
        np.random.random(size=n_samples) < 0.5
    )  # Random choice of the glitch amplitude sign.
    amp[sign] *= -1

    return amp


def betas_dist_set(n_samples=10, beta_set=[0.001, 100]):
    try:
        betas = np.linspace(beta_set[0], beta_set[1], n_samples)
    except TypeError or IndexError:
        # TODO: Maybe add sys.ext?
        betas = np.linspace(beta_set, n_samples)

    return betas


def betas_dist_exponential(n_samples=10, scale=4, seed=None):
    np.random.seed(seed)

    betas = np.random.exponential(float(scale), n_samples)

    return betas