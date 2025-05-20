# search.py
"""
Identify transient GW events (triggers) in LISA data.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
from scipy.signal.windows import tukey
from gwpy.timeseries import TimeSeriesDict, TimeSeriesList, TimeSeries
import matplotlib.pyplot as plt

GATE_SEGLEN = 300
GATE_RMS = 4.5
REFINE_TSPAN = 18000

PATH_cd = os.getcwd()
PATH_qfiles = os.path.join(PATH_cd, 'q_transform/q_files/')


def read_tdi(file, tdi):
    """Read TDI time series data.
    """
    data_dict = TimeSeriesDict.read(file)

    tdi_dict = {'X': data_dict['X'],
                'Y': data_dict['Y'],
                'Z': data_dict['Z'],
                'A': (data_dict['Z'] - data_dict['X']) / 2 ** 0.5,
                'E': (data_dict['X'] - 2*data_dict['Y'] + data_dict['Z']) / 6**0.5,
                'EV': (data_dict['Y'] - 2 * data_dict['Z'] + data_dict['X']) / 6 ** 0.5,
                'T': (data_dict['X'] + data_dict['Y'] + data_dict['Z']) / 3**0.5}

    PATH_tdi = PATH_qfiles + f'/tdi{tdi}'
    try:
        os.mkdir(PATH_tdi)
        print("Directory doesn't exists, creating: ", PATH_tdi)
    except OSError as error:
        print('Directory already exists')

    data = tdi_dict[tdi]
    data.name = tdi
    return data, PATH_tdi


def segment_data(data, segment_length):
    """section data into segment_length (given in seconds) segments
    """

    dp = int(segment_length / 5)  # Beth here change this
    # dp = int(segment_length / 0.25)

    tdi_dict = {'segment': {}}
    i, j = 0, 0
    while i + dp <= len(data):
        if i + dp < len(data):
            idx_start, idx_end = i, i + dp
            small_dict = data[idx_start: idx_end]
            tdi_dict['segment'][j] = small_dict
            i = i + int(dp / 2)
        else:
            idx_start, idx_end = i, -1
            small_dict = data[idx_start: idx_end]
            tdi_dict['segment'][j] = small_dict
            break
        j += 1
    if i + dp > len(data):
        idx_start, idx_end = i, -1
        small_dict = data[idx_start: idx_end]
        tdi_dict['segment'][j] = small_dict

    return tdi_dict


def _gate(data, seglen, rms_factor):
    """Apply gating to time series data. Any segments
    crossing the amplitude threshold are zeroed out, with
    inverse Tukey roll-offs on either end.
    """
    data_gated = data.copy()
    thres = (data.value**2).mean()**0.5 * rms_factor
    
    # Creating gated data for ASD
    nperseg = int(seglen * data.sample_rate.value)
    nseg = np.floor(data.size / nperseg).astype(int)
    ngates = 0
    for i in range(nseg):
        start_ind = i * nperseg
        if i == nseg - 1:
            end_ind = -1
        else:
            end_ind = start_ind + nperseg 
        segment = data[start_ind:end_ind]

        if any(abs(segment.value) > thres):
            data_gated[start_ind:end_ind] = np.zeros(segment.size)
            ngates += 1

            # Tukey windowing
            if start_ind != 0:
                window_l = tukey(nperseg, alpha=0.5)[-nperseg//4:]
                data_gated[start_ind-nperseg//4:start_ind] *= window_l
            if end_ind != -1:
                window_r = tukey(nperseg, alpha=0.5)[:nperseg//4]
                data_gated[end_ind:end_ind+nperseg//4] *= window_r

    per_gated = round(ngates * seglen / data.duration.value * 100, 3)

    return data_gated


def _get_qs(qmin, qmax, mismatch):
    """Get the needed Q-values (function from GWpy).
    """
    cumum = np.log(qmax / qmin) / 2**0.5
    deltam = 2 * (mismatch / 3 )**0.5
    nplanes = int(max(np.ceil(cumum / deltam), 1))
    dq = cumum / nplanes

    return qmin * np.exp(2**0.5 * dq * (np.arange(nplanes) + 0.5))


def _generate_triggers(data, qmin, qmax, fmin, fmax, snr_thres, 
                       mismatch, tmin, tmax, verbose=True):
    """Generate the Q-transform of the data and return
    significant triggers.
    """

    energy_thres = snr_thres**2 + 1

    qs = _get_qs(qmin, qmax, mismatch)
    for q in qs:
        qtransform = data.q_gram(
            qrange=(q, q),  # force one Q-plane at a time
            frange=(fmin, fmax),
            snrthresh=2,
            mismatch=mismatch,
            norm='median')

        # Read tile params
        tile_time = np.array(qtransform['time'])
        tile_freq = np.array(qtransform['frequency'])
        tile_dur = np.array(qtransform['duration'])
        tile_band = np.array(qtransform['bandwidth'])
        # normalize by median so need factor of log(2) here
        tile_energy = np.array(qtransform['energy']) * np.log(2)
        
        # Apply threshold
        above_thres = tile_energy > energy_thres
        tile_time = tile_time[above_thres]
        tile_freq = tile_freq[above_thres]
        tile_dur = tile_dur[above_thres]
        tile_band = tile_band[above_thres]
        tile_energy = tile_energy[above_thres]

        # Subtract expectation and compute SNR
        subexp = tile_energy - 1
        subexp[subexp < 0] = 0
        tile_snr = np.sqrt(subexp)

        tiles = np.column_stack((tile_time, tile_freq, tile_dur, tile_band, 
                                 np.repeat(q, tile_time.size), tile_energy, 
                                 tile_snr))
        
        # Only include triggers within time range
        if all((tmin, tmax)):
            tiles = tiles[(tiles[:, 0] > tmin) & (tiles[:, 0] < tmax)]

        if q == qs[0]:
            raw_triggers = tiles
        else:
            raw_triggers = np.vstack((raw_triggers, tiles))

    return raw_triggers


def _cluster_triggers(triggers, deltat):
    """Cluster nearby triggers in time and return the
    loudest trigger in each cluster.
    """
    # Sort triggers by start time
    start_times = triggers[:, 0] - triggers[:, 2]/2
    triggers_sorted = triggers[start_times.argsort()]

    # Clustering routine
    clusters = []
    last = len(triggers) - 1
    for ind, trig in enumerate(triggers_sorted):

        if ind == 0:
            cl = np.array([trig])
        else:
            t1 = trig[0] - trig[2]/2
            t2 = max(cl[:, 0] + cl[:, 2]/2)
            dt = t1 - t2

            if dt > deltat:
                clusters.append(cl)
                cl = np.array([trig])

            else:
                cl = np.vstack((cl, trig))

        if ind == last:
            clusters.append(cl)

    # Select loudest trigger from each cluster
    max_triggers = []
    for cluster in clusters:
        indmax = cluster[:, -1].argmax()
        max_triggers.append(cluster[indmax])

    for i in range(len(max_triggers) - 1):
        curr, proc = max_triggers[i], max_triggers[i+1]
        t1, tdur1 = curr[0], curr[2]
        t2, tdur2 = proc[0], proc[2]
        dt = (t2 + tdur2/2) - (t1 - tdur1/2)

    return np.array(max_triggers)


def _refine_triggers(data, triggers, tspan, qmin, qmax, fmin, fmax, 
                     snr_thres, mismatch):
    """Refine the trigger parameters with a higher resolution Q-transform.
    """
    trigger_times = triggers[:, 0]
    for trigger_time in trigger_times:
        segment = data.crop(trigger_time - tspan/2, trigger_time + tspan/2)
        raw_triggers = _generate_triggers(segment, qmin, qmax, fmin, fmax, snr_thres,
                                          mismatch, tmin=None, tmax=None, verbose=False)
        
        # Max-SNR trigger
        indmax = raw_triggers[:, -1].argmax()
        max_trigger = raw_triggers[indmax]

        if trigger_time == trigger_times[0]:
            refined_triggers = max_trigger
        else:
            refined_triggers = np.vstack((refined_triggers, max_trigger))

    return refined_triggers


def run_burst_search(data, qmin, qmax, fmin, fmax, snr_thres, mismatch, deltat, 
                     refine_mismatch=None, asd_fftlen=None, asd_overlap=None, 
                     wfdur=None, tmin=None, tmax=None, out_label=None, 
                     save_triggers=False, save_conditioned=False):
    """Search for burst signals using the Q-transform, returns a set of triggers 
    corresponding to clustered time-frequency tiles above the SNR threshold.
    """
    if all((asd_fftlen, asd_overlap, wfdur)):
        # Whiten using ASD of gated time series
        data_gated = _gate(data, seglen=GATE_SEGLEN, rms_factor=GATE_RMS)
        asd = data_gated.asd(fftlength=asd_fftlen, overlap=asd_overlap,
                             window='hann', method='median')
        if save_conditioned:
            asd.write(f'{out_label}_asd_gated.txt')
            data_gated.write(f'{out_label}_data_gated.h5', overwrite=True)

        # Normalize so that PSD has unit mean
        data = data.whiten(asd=asd, fduration=wfdur) / (2*data.dt.value)**0.5
        if save_conditioned:
            data.write(f'{out_label}_data_whitened.h5', overwrite=True)
        data_whitened = data
    else:
        # Taper the start and end of the data to remove edge artifacts
        data *= tukey(data.size, alpha=0.001)

    # Generate triggers from Q-transform
    raw_triggers = _generate_triggers(data, qmin, qmax, fmin, fmax,
                                      snr_thres, mismatch, tmin, tmax)

    if len(raw_triggers) == 0:
        return None, None, data_whitened, asd

    # Run clustering algorithm
    triggers = _cluster_triggers(raw_triggers, deltat)

    # Refine the trigger parameters
    if refine_mismatch:
        refined_triggers = _refine_triggers(data, triggers, REFINE_TSPAN, qmin, qmax,
                                            fmin, fmax, snr_thres, refine_mismatch)
            
    if save_triggers:
        np.savetxt(f'{out_label}_triggers.dat', triggers)
        np.savetxt(f'{out_label}_triggers_unclustered.dat', raw_triggers)
        if refine_mismatch:
            np.savetxt(f'{out_label}_triggers_refined.dat', refined_triggers)

    if refine_mismatch:
        return triggers, raw_triggers, refined_triggers
    else:
        return triggers, raw_triggers, data_whitened, asd


def main():
    """Command line tool for finding triggers.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file',
                        help='Data file to process', type=str)
    parser.add_argument('--tdi',
                        help='Which TDI variable to use', type=str)
    parser.add_argument('--asd-fftlen',
                        help='FFT length for ASD estimation', type=int)
    parser.add_argument('--asd-overlap',
                        help='FFT overlap for ASD estimation', type=int)
    parser.add_argument('--wfdur',
                        help='Whitening filter duration', type=int)
    parser.add_argument('--qmin',
                        help='Minimum Q-value', type=float)
    parser.add_argument('--qmax',
                        help='Maximum Q-value', type=float)
    parser.add_argument('--fmin',
                        help='Minimum frequency', type=float)
    parser.add_argument('--fmax',
                        help='Maximum frequency', type=float)
    parser.add_argument('--snr-thres',
                        help='Minimum tile SNR threshold', type=float)
    parser.add_argument('--mismatch',
                        help='Maximum tile mismatch', type=float)
    parser.add_argument('--deltat',
                        help='Clustering time interval', type=float)
    parser.add_argument('--refine-mismatch',
                        help='Refinement mismatch', type=float)
    parser.add_argument('--tmin', 
                        help='Output segment start time', type=float)
    parser.add_argument('--tmax',
                        help='Output segment end time', type=float)
    parser.add_argument('--label',
                        help='Job label', type=str)
    parser.add_argument('--segments',
                        help='Segment the input dat before searching', type=bool)
    args = parser.parse_args()

    data, PATH_tdi = read_tdi(PATH_cd + args.data_file, args.tdi)

    print('in choppy')
    # Chop data into 24 hour segments
    tdi_in_sections = segment_data(data, segment_length=24*3600)  # 24 hour segments

    ref_triggers_arr, triggers_arr, raw_triggers_arr = np.zeros((1, 7)), np.empty((1, 7)), np.zeros((1, 7))
    ref_trigger_tracker, trigger_tracker, white_data_segments = [], [], []
    i = 0
    asd_data, checkout_segments = [], []

    for seg in range(len(tdi_in_sections['segment'])):
        data_update = tdi_in_sections['segment'][seg]
        t0 = data_update.times.value[0]
        dt = np.abs(data_update.times.value[1] - data_update.times.value[0])

        win = tukey(data_update.size, alpha=0.0001)
        data_update = TimeSeries(data_update * win, t0=t0, dt=dt)

        # Run segment through burst search
        triggers, raw_triggers, whitened_data, asd = \
            run_burst_search(
                data_update, qmin=args.qmin, qmax=args.qmax, fmin=args.fmin, fmax=args.fmax,
                snr_thres=args.snr_thres, mismatch=args.mismatch, deltat=args.deltat,
                refine_mismatch=args.refine_mismatch, asd_fftlen=args.asd_fftlen,
                asd_overlap=args.asd_overlap, wfdur=args.wfdur, tmin=args.tmin, tmax=args.tmax,
                out_label=args.label)
        if triggers is not None:
            raw_triggers_arr = np.vstack((raw_triggers_arr, raw_triggers))
            for k in range(len(triggers)):
                t = triggers[k]
                if t[0] not in trigger_tracker:
                    triggers_arr = np.vstack((triggers_arr, t))
                    trigger_tracker.append(t[0])
                else:
                    idx = np.where(t[0] == trigger_tracker)[0]

        white_data_segments.append(whitened_data)
        checkout_segments.append(seg)
        asd_data.append(asd)
        i += 1

    # remove the duplicates of triggers and ones that might be from the same trigger
    start_times = triggers_arr[:, 0] - triggers_arr[:, 2] / 2
    triggers_sorted = triggers_arr[start_times.argsort()]
    idx_remove = []
    for i in range(len(triggers_sorted)-1):
        t1, t1_dur = triggers_sorted[:, 0][i], triggers_sorted[:, 2][i]
        t2, t2_dur = triggers_sorted[:, 0][i + 1], triggers_sorted[:, 2][i + 1]

        end_t1 = t1 + t1_dur / 2
        start_t2 = t2 - t2_dur / 2

        diff = start_t2 - end_t1
        if np.abs(diff) <= 250:
            if triggers_sorted[:, -1][i] > triggers_sorted[:, -1][i+1]:
                idx_remove.append(i+1)
            else:
                idx_remove.append(i)
    triggers_arr = np.delete(triggers_sorted, idx_remove, 0)

    if len(white_data_segments) == 1:
        white_data_segments[0].write(f'{PATH_tdi}/{args.label}_all_data_whitened.h5', overwrite=True)
    else:
        for i in range(1, len(white_data_segments)):
            if i in checkout_segments:
                prev = white_data_segments[i - 1]
                curr = white_data_segments[i]
                if prev[int(len(prev.times) / 2):].shape == curr[:int(len(curr.times) / 2)].shape:
                    new = (prev[int(len(prev.times) / 2):] + curr[:int(len(curr.times) / 2)]) / 2
                    new_copy = new.copy()
                    if i > 1:
                        keep = keep.copy()
                        keep.append(new_copy, inplace=True)
                    elif i == len(white_data_segments):
                        break
                    elif i == 1:
                        keep = prev[:int(len(prev.times) / 2)].copy()
                        keep.append(new_copy, inplace=True)
                else:
                    amt = int(len(prev.times) / 2)
                    new = (prev[amt:] + curr[:amt]) / 2
                    new_copy = new.copy()
                    keep = keep.copy()
                    keep.append(new_copy, inplace=True)
                    keep = keep.copy()
                    keep.append(curr[amt:-1], inplace=True)

        keep.write(f'{PATH_tdi}/{args.label}_all_data_whitened.h5', overwrite=True)

    np.savetxt(f'{PATH_tdi}/{args.label}_triggers.dat', triggers_arr[1:])
    np.savetxt(f'{PATH_tdi}/{args.label}_triggers_unclustered.dat', raw_triggers_arr[1:])


if __name__ == "__main__":

    try:
        os.mkdir(PATH_qfiles)
    except OSError as error:
        print(error)

    print('Starting Q')
    main()

