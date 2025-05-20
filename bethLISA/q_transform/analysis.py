"""
Help with visualization and analysis of q-transform
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gwpy.timeseries import TimeSeriesDict
import matplotlib.cm as cm
import matplotlib.colors as col
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from KyeLDC.Code.KyeLISAModule import analysis as kla
from lisaHTI.search import _generate_triggers, _cluster_triggers
# from testing import *
from astropy import units
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from prettytable import PrettyTable
import importlib
import itertools
import h5py
import os

cmap_name = 'YlGnBu'
cmap = cm.get_cmap(cmap_name)

PATH_cd = os.getcwd()
PATH_lgs = os.path.join(PATH_cd, 'lisa_glitch_simulation/')
PATH_qtrans = os.path.join(PATH_cd, 'q_transform/')
print('PATH_qtrans', PATH_qtrans)
PATH_qfiles = os.path.join(PATH_cd, 'q_transform/q_files/')
PATH_blisa = os.path.abspath(os.path.join(PATH_cd, os.pardir))  # PATH to lisa_glitch_simulation directory
PATH_tdi = os.path.join(PATH_lgs, 'final_tdi_outputs/')


# Plotting Q-transform tiles
def color_map_color(value, vmin=1, vmax=10):
    norm = col.Normalize(vmin=vmin, vmax=vmax)
    rgb = cmap(norm(abs(value)))[:3]
    color = col.rgb2hex(rgb)

    return color


def get_cmap_mappable(vmin=1, vmax=10):
    norm = col.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    return sm


def plot_trigger(axs, t, f, dt, df, snr, vmin=1, vmax=10):
    c = color_map_color(snr, vmin=vmin, vmax=vmax)
    tile = Rectangle((t - dt / 2, f - df / 2), dt, df, color=c)
    axs.add_artist(tile)


def plot_all(times, freqs, dt, df, snr, trange=None, frange=None, title='', duration_bar=False, bw_bar=False,
             ax=None, focus_freq=None, focus_time=None, focus_bw=None, focus_dur=None, focus_snr=None):

    # Plot Q-transform tiles
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    plt.suptitle(title)

    fmin, fmax = np.min(freqs), np.max(freqs)
    tmin, tmax = np.min(times), np.max(times)

    vmin, vmax = 4.5, 40
    for i in range(times.shape[0]):
        plot_trigger(ax, times[i], freqs[i], dt[i], df[i], snr[i], vmin=vmin, vmax=vmax)

    if trange == None:
        ax.set_xlim((tmin, tmax))
    else:
        ax.set_xlim(trange)
    if frange == None:
        ax.set_ylim((fmin, fmax))
    else:
        ax.set_ylim(frange)
    ax.set_yscale("log")

    # Colorbar
    sm = get_cmap_mappable(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(sm, ax=ax, shrink=1, aspect=40, label='SNR', pad=0, location='top')

    # plot the duration
    if duration_bar:
        print('focus_time', focus_time)
        plt.hlines(y=focus_freq - focus_bw/2, xmin=focus_time - focus_dur/2, xmax=focus_time + focus_dur/2,
                   color='r', linestyle='--')
        plt.hlines(y=focus_freq + focus_bw/2, xmin=focus_time - focus_dur/2, xmax=focus_time + focus_dur/2,
                   color='r', linestyle='--')
    if bw_bar:
        print('focus_freq', focus_freq)
        plt.vlines(x=focus_time - focus_dur/2, ymin=focus_freq - focus_bw/2, ymax=focus_freq + focus_bw/2,
                   color='r', linestyle='--')
        plt.vlines(x=focus_time + focus_dur/2, ymin=focus_freq - focus_bw/2, ymax=focus_freq + focus_bw/2,
                   color='r', linestyle='--')
        print('focus_snr', focus_snr)
        ax.scatter(focus_time, focus_freq, marker='s', color='purple')

    plt.show()


# Analysis + Last steps of q-transform detection
def label_glitch(tdiX, tdiY, tdiZ, tdiT=None, tdiE=None, tdiEV=None, t_range=None):
    """compare all 3 tdi channels and check through the T channel,
    if a trigger shows in 2/3 then label as a glitch
    if a trigger shows in tdiT then also label as a glitch
    return the time of glitch
    if tdiE and/or tdiA are given check these as well"""
    tdiX_times, tdiY_times, tdiZ_times = tdiX[:, 0], tdiY[:, 0], tdiZ[:, 0]
    tdiX_durs, tdiY_durs, tdiZ_durs = tdiX[:, 2], tdiY[:, 2], tdiZ[:, 2]
    tdiX_freq, tdiY_freq, tdiZ_freq = tdiX[:, 1], tdiY[:, 1], tdiZ[:, 1]
    tdiX_snrs, tdiY_snrs, tdiZ_snrs = tdiX[:, -1], tdiY[:, -1], tdiZ[:, -1]
    glitches, gw = {}, {}
    glitchXY, glitchXZ, glitchYZ = [], [], []
    glitchXY_durs, glitchXZ_durs, glitchYZ_durs = [], [], []
    glitchXY_freqs, glitchXZ_freqs, glitchYZ_freqs = [], [], []
    glitchXY_snrs, glitchXZ_snrs, glitchYZ_snrs = [], [], []
    glitch_channels = []

    if t_range is None:
        for i in range(len(tdiX_times)):
            trigX = tdiX_times[i]
            if trigX in tdiY_times and trigX in tdiZ_times:
                gw['GW'] = trigX
            elif trigX in tdiY_times:
                idx_snr = np.where(tdiY_times == trigX)[0]
                glitchXY.append(trigX)
                glitchXY_durs.append(tdiX_durs[i])
                snrs = [tdiX_snrs[i], tdiY_snrs[idx_snr][0]]
                glitchXY_snrs.append(snrs)
            elif trigX in tdiZ_times:
                idx_snr = np.where(tdiZ_times == trigX)[0]
                glitchXZ.append(trigX)
                glitchXZ_durs.append(tdiX_durs[i])
                snrs = [tdiX_snrs[i], tdiZ_snrs[idx_snr][0]]
                glitchXZ_snrs.append(snrs)
        for j in range(len(tdiZ_times)):
            trigZ = tdiZ_times[j]
            if trigZ in tdiY_times:
                idx_snr = np.where(tdiY_times == trigZ)[0]
                glitchYZ.append(trigZ)
                glitchYZ_durs.append(tdiZ_durs[j])
                snrs = [tdiY_snrs[idx_snr][0], tdiZ_snrs[j]]
                glitchYZ_snrs.append(snrs)

        glitches['XY'], glitches['XZ'], glitches['YZ'] = np.array(glitchXY), np.array(glitchXZ), np.array(glitchYZ)
        glitches['XY dur'], glitches['XZ dur'], glitches['YZ dur'] = \
            np.array(glitchXY_durs), np.array(glitchXZ_durs), np.array(glitchYZ_durs)
        glitches['XY snr'], glitches['XZ snr'], glitches['YZ snr'] = \
            np.array(glitchXY_snrs), np.array(glitchXZ_snrs), np.array(glitchYZ_snrs)

        if tdiE is not None:
            glitches['E'], glitches['E dur'], glitches['E snr'] = \
                tdiE[:, 0], tdiE[:, 2],  tdiE[:, -1]
        if tdiEV is not None:
            glitches['EV'], glitches['EV dur'], glitches['EV snr'] = \
                tdiEV[:, 0], tdiEV[:, 2],  tdiEV[:, -1]

        if tdiT is None:
            return glitches, gw
        else:
            glitches['T'], glitches['T dur'], glitches['T snr'] = tdiT[:, 0], tdiT[:, 2], tdiT[:, -1]
            return glitches, gw

    else:
        for i in range(len(tdiX_times)):
            trigX = tdiX_times[i]

            tdiY_close_idx = np.where((np.abs(tdiY_times - trigX) >= 0.0) &
                                      (np.abs(tdiY_times - trigX) <= t_range))[0]
            tdiZ_close_idx = np.where((np.abs(tdiZ_times - trigX) >= 0.0) &
                                      (np.abs(tdiZ_times - trigX) <= t_range))[0]

            tdiY_close_idx = np.where((np.abs(tdiY_times - trigX) == 0.0))[0]
            tdiZ_close_idx = np.where((np.abs(tdiZ_times - trigX) == 0.0))[0]

            tdiY_close = tdiY_times[tdiY_close_idx]
            tdiZ_close = tdiZ_times[tdiZ_close_idx]

            if trigX in tdiY_times and trigX in tdiZ_times:
                gw['GW'] = trigX
            elif (trigX in tdiY_times):  # or len(tdiY_close) > 0:
                if len(tdiY_close) > 0:
                    for a in range(len(tdiY_close)):
                        idx_snr = np.where(tdiY_close[a] == trigX)[0]
                        if len(tdiY_snrs[idx_snr]) == 0:
                            snrs = [tdiX_snrs[i]]
                        else:
                            snrs = [tdiX_snrs[i], tdiY_snrs[idx_snr][0]]
                        glitchXY_snrs.append(snrs)
                        glitchXY.append(tdiY_close[a])
                        glitchXY_durs.append(tdiX_durs[i])
                        glitchXY_freqs.append(tdiX_freq[i])
                else:
                    idx_snr = np.where(tdiY_times == trigX)[0]
                    if len(tdiY_snrs[idx_snr]) == 0:
                        snrs = [tdiX_snrs[i]]
                    else:
                        snrs = [tdiX_snrs[i], tdiY_snrs[idx_snr][0]]
                    glitchXY_snrs.append(snrs)
                    glitchXY.append(trigX)
                    glitchXY_durs.append(tdiX_durs[i])
                    glitchXY_freqs.append(tdiX_freq[i])

            elif trigX in tdiZ_times:  # or len(tdiZ_close) > 0:
                if len(tdiZ_close) > 0:
                    for a in range(len(tdiZ_close)):
                        idx_snr = np.where(tdiZ_close[a] == trigX)[0]
                        if len(tdiZ_snrs[idx_snr]) == 0:
                            snrs = [tdiX_snrs[i]]
                        else:
                            snrs = [tdiX_snrs[i], tdiZ_snrs[idx_snr][0]]
                        glitchXZ_snrs.append(snrs)
                        glitchXZ.append(tdiZ_close[a])
                        glitchXZ_durs.append(tdiX_durs[i])
                        glitchXZ_freqs.append(tdiX_freq[i])
                else:
                    idx_snr = np.where(tdiZ_times == trigX)[0]
                    if len(tdiZ_snrs[idx_snr]) == 0:
                        snrs = [tdiX_snrs[i]]
                    else:
                        snrs = [tdiX_snrs[i], tdiZ_snrs[idx_snr][0]]
                    glitchXZ_snrs.append(snrs)
                    glitchXZ.append(trigX)
                    glitchXZ_durs.append(tdiX_durs[i])
                    glitchXZ_freqs.append(tdiX_freq[i])

        for j in range(len(tdiZ_times)):
            trigZ = tdiZ_times[j]
            tdiY_close = tdiY_times[np.where((np.abs(tdiY_times - trigZ) > 0.0) &
                                             (np.abs(tdiY_times - trigZ) <= t_range))[0]]
            if trigZ in tdiY_times:  # or len(tdiY_close) > 0:
                if len(tdiY_close):
                    for i in range(len(tdiY_close)):
                        idx_snr = np.where(tdiY_close[i] == trigZ)[0]
                        if len(tdiY_snrs[idx_snr]) == 0:
                            snrs = [tdiZ_snrs[j]]
                        else:
                            snrs = [tdiY_snrs[idx_snr][0], tdiZ_snrs[j]]
                        glitchYZ_snrs.append(snrs)
                        glitchYZ.append(tdiY_close[i])
                        glitchYZ_durs.append(tdiZ_durs[j])  # change
                        glitchYZ_freqs.append(tdiZ_freq[j])
                else:
                    idx_snr = np.where(tdiY_times == trigZ)[0]
                    if len(tdiY_snrs[idx_snr]) == 0:
                        snrs = [tdiZ_snrs[j]]
                    else:
                        snrs = [tdiY_snrs[idx_snr][0], tdiZ_snrs[j]]
                    glitchYZ_snrs.append(snrs)
                    glitchYZ.append(trigZ)
                    glitchYZ_durs.append(tdiZ_durs[j])
                    glitchYZ_freqs.append(tdiZ_freq[j])

        glitches['XY'], glitches['XZ'], glitches['YZ'] = \
            np.array(glitchXY), np.array(glitchXZ), np.array(glitchYZ)
        glitches['XY dur'], glitches['XZ dur'], glitches['YZ dur'] = \
            np.array(glitchXY_durs), np.array(glitchXZ_durs), np.array(glitchYZ_durs)
        glitches['XY freq'], glitches['XZ freq'], glitches['YZ freq'] = \
            np.array(glitchXY_freqs), np.array(glitchXZ_freqs), np.array(glitchYZ_freqs)
        glitches['XY snr'], glitches['XZ snr'], glitches['YZ snr'] = \
            glitchXY_snrs, glitchXZ_snrs, glitchYZ_snrs

        if tdiE is not None:
            glitches['E'], glitches['E dur'], glitches['E freq'], glitches['E snr'] = \
                tdiE[:, 0], tdiE[:, 2], tdiE[:, 1], tdiE[:, -1]
        if tdiEV is not None:
            glitches['EV'], glitches['EV dur'], glitches['EV freq'], glitches['EV snr'] = \
                tdiEV[:, 0], tdiEV[:, 2], tdiEV[:, 1], tdiEV[:, -1]

        if tdiT is None:
            return glitches, gw
        else:
            glitches['T'], glitches['T dur'], glitches['T freq'], glitches['T snr'] = \
                tdiT[:, 0], tdiT[:, 2], tdiT[:, 1], tdiT[:, -1]
            return glitches, gw


def find_amplitudes(ch, glitch_times12, glitch_durs12, glitch_freqs12=None, glitch_snrs12=None,
                    tdi1=None, tdi2=None, need_amp=False):
    """glitch_times is a list with 'XY', 'XZ', 'YZ' or 'T' glitch times, want to find the amplitude of each glitch
    if tdi2=None then we've passed through glitches of the 'T' channel
    glitch_durs12 is a list of glitch durations
    return a dictionary of glitch times their respective amplitudes"""

    if need_amp:
        glitch_dict = {}
        glitch_count = 0
        for i in range(len(glitch_times12)):
            t_start, t_end = glitch_times12[i] - glitch_durs12[i], glitch_times12[i] + glitch_durs12[i]
            glitch_dur = glitch_durs12[i]
            glitch_freq, glitch_snr = glitch_freqs12[i], glitch_snrs12[i]
            idx_start = np.abs(tdi1.times.value - t_start).argmin()
            idx_end = np.abs(tdi1.times.value - t_end).argmin()
            if idx_start == idx_end:
                idx_end += 1
            if tdi2 is None:
                amp_idx = np.argmax(np.abs(np.asarray(tdi1[idx_start:idx_end])))
                amplitude = tdi1[idx_start:idx_end][amp_idx]
                t_glitch = tdi1.times[idx_start:idx_end][amp_idx]
            else:
                max1_idx = np.argmax(np.abs(np.asarray(tdi1[idx_start:idx_end])))
                max2_idx = np.argmax(np.abs(np.asarray(tdi2[idx_start:idx_end])))
                max1_amp, max2_amp = tdi1[idx_start:idx_end][max1_idx], tdi2[idx_start:idx_end][max2_idx]
                if max1_amp >= max2_amp:
                    amplitude = max1_amp
                    t_glitch = tdi1.times[idx_start:idx_end][max1_idx]
                else:
                    amplitude = max2_amp
                    t_glitch = tdi2.times[idx_start:idx_end][max2_idx]

            glitch_dict[glitch_count] = {'glitch time': t_glitch.value, 'amplitude': amplitude,
                                         'glitch dur': glitch_dur, 'glitch freq': glitch_freq, 'glitch snr': glitch_snr}

            glitch_count += 1
        return glitch_dict

    else:
        glitch_dict = {}
        glitch_count = 0
        for i in range(len(glitch_times12)):
            glitch_dict[glitch_count] = {'glitch time': glitch_times12[i],
                                         'glitch dur': glitch_durs12[i],
                                         'glitch freq': glitch_freqs12[i],
                                         'glitch snr': glitch_snrs12[i]}
            glitch_count += 1

        return glitch_dict


def find_false_positives(glitches_e, glitches_ev, glitches_t, glitches_xy, glitches_xz, glitches_yz, white_data):
    """
    try to identify the false positives
    """

    xy_times = get_list_of_key(glitches_xy, ['glitch time'])
    xz_times = get_list_of_key(glitches_xz, ['glitch time'])
    yz_times = get_list_of_key(glitches_yz, ['glitch time'])
    t_times = get_list_of_key(glitches_t, ['glitch time'])
    e_times = get_list_of_key(glitches_e, ['glitch time'])
    ev_times = get_list_of_key(glitches_ev, ['glitch time'])

    times_to_check = [xy_times, xz_times, yz_times]
    glitch_files = [glitches_xy, glitches_xz, glitches_yz]
    times_to_compare_against = [t_times, e_times, ev_times]
    VARS = ['XY', 'XZ', 'YZ', 'T', 'E', 'EV']

    keep_glitches_all, kicked_out_all, kicked_out_n = {}, {}, 0
    for i in range(len(times_to_check)):
        times = times_to_check[i]
        glitch_file = glitch_files[i]
        keep_glitches, kicked_out = {}, {}
        for t in times:
            idx = np.where(t == np.asarray(times))[0]
            if t in e_times or t in ev_times:
                # keep time because it was found in one or more of the channels sensitive* to glitches
                glitch_info = glitch_file[idx[0]]
                keep_glitches[t] = {'glitch time': glitch_info['glitch time'], 'glitch dur': glitch_info['glitch dur'],
                                 'glitch freq': glitch_info['glitch freq'], 'glitch snr': glitch_info['glitch snr']}
            else:
                glitch_info = glitches_t[idx[0]]
                kicked_out[t] = {'glitch time': glitch_info['glitch time'], 'glitch dur': glitch_info['glitch dur'],
                                    'glitch freq': glitch_info['glitch freq'], 'glitch snr': glitch_info['glitch snr']}
        keep_glitches_all[VARS[i]] = keep_glitches
        kicked_out_all[VARS[i]] = kicked_out

    keep_glitches, kicked_out = {}, {}
    for t in t_times:
        idx = np.where(t == np.asarray(t_times))[0]
        if t in e_times or t in ev_times or t in xy_times or t in xz_times or t in yz_times:
            # keep time because it was found in one or more of the channels sensitive* to glitches
            glitch_info = glitches_t[idx[0]]
            keep_glitches[t] = {'glitch time': glitch_info['glitch time'], 'glitch dur': glitch_info['glitch dur'],
                                'glitch freq': glitch_info['glitch freq'], 'glitch snr': glitch_info['glitch snr']}
        else:
            glitch_info = glitches_t[idx[0]]
            kicked_out[t] = {'glitch time': glitch_info['glitch time'], 'glitch dur': glitch_info['glitch dur'],
                             'glitch freq': glitch_info['glitch freq'], 'glitch snr': glitch_info['glitch snr']}
    keep_glitches_all['T'] = keep_glitches
    kicked_out_all['T'] = kicked_out

    return keep_glitches_all, kicked_out_all


def find_corresponding_glitch(glitch_real, glitches_found, coin_window, channels=None, just_times=False):
    """find the corresponding glitches for found_glitches in real_glitches"""
    glitch_times_real, glitch_amps_real = glitch_real['time'], glitch_real['level']
    glitch_corr, glitch_false = {}, {}
    times, amps = [], []
    tracking, rev_dict = {}, {}
    j, k = 0, 0
    kicked_out_count = 0

    if not just_times:

        sorting_found_glitches = sorted(glitches_found.items(), key=lambda x: x[1]['glitch time'])
        sorted_glitches_found = dict(sorting_found_glitches)

        for key in sorted_glitches_found.keys():
            time = sorted_glitches_found[key]['glitch time']
            freq = sorted_glitches_found[key]['glitch freq']
            snrs = sorted_glitches_found[key]['glitch snr']
            durs = sorted_glitches_found[key]['glitch dur']

            idx_close = np.abs(glitch_times_real - time).argmin()
            time_corr = glitch_times_real[idx_close]
            amp_corr = glitch_amps_real[idx_close]
            diff = time - time_corr

            if np.abs(diff) > coin_window:
                glitch_false[key] = {"time trigg": time, "time corr": time_corr,
                                     'freq': freq, "diff": diff, "channels": channels, 'snr': snrs, 'dur': durs}
            else:
                glitch_corr[key] = {"time trigg": time, "time corr": time_corr,
                                    "amp corr": amp_corr, 'freq': freq,
                                    "diff": diff, "idx_corr": idx_close,
                                    "channels": channels, 'snr': snrs, 'dur': durs}
                tracking[key] = time_corr
                rev_dict.setdefault(time_corr, set()).add(key)

        for dupes in rev_dict.items():
            if len(dupes[1]) > 1:
                times = []
                for key in dupes[1]:
                    times.append(glitch_corr[key]["time trigg"])
                    time_corr = glitch_corr[key]["time corr"]
                idx_close = np.abs(times - time_corr).argmin()
                new_dupes = list(dupes[1])
                del new_dupes[idx_close]
                del times[idx_close]

                i = 0
                for key in new_dupes:
                    freq = glitch_corr[key]['freq']
                    diff = glitch_corr[key]['diff']
                    snrs = glitch_corr[key]['snr']
                    durs = glitch_corr[key]['dur']
                    del glitch_corr[key]
                    glitch_false[key] = {"time trigg": times[i], "time corr": time_corr,
                                         'freq': freq, "diff": diff, "channels": channels, 'snr': snrs, 'dur': durs}
                    i += 1

        return glitch_corr, len(glitch_false.keys()), glitch_false

    else:
        sorting_found_glitches = sorted(glitches_found, key=lambda x: x)
        sorted_glitches_found = sorting_found_glitches

        for key in range(len(sorted_glitches_found)):
            time = sorted_glitches_found[key]

            idx_close = np.abs(glitch_times_real - time).argmin()
            time_corr = glitch_times_real[idx_close]
            amp_corr = glitch_amps_real[idx_close]
            diff = time - time_corr

            if np.abs(diff) > coin_window:
                glitch_false[key] = {"time trigg": time, "time corr": time_corr,
                                     "diff": diff, "channels": channels}
            else:
                glitch_corr[key] = {"time trigg": time, "time corr": time_corr, "amp corr": amp_corr,
                                    "diff": diff, "idx_corr": float(idx_close), "channels": channels}
                tracking[key] = time_corr
                rev_dict.setdefault(time_corr, set()).add(key)

        for dupes in rev_dict.items():
            if len(dupes[1]) > 1:
                times = []
                for key in dupes[1]:
                    times.append(glitch_corr[key]["time trigg"])
                    time_corr = glitch_corr[key]["time corr"]
                idx_close = np.abs(times - time_corr).argmin()
                new_dupes = list(dupes[1])
                del new_dupes[idx_close]
                del times[idx_close]

                i = 0
                for key in new_dupes:
                    diff = glitch_corr[key]['diff']
                    del glitch_corr[key]
                    glitch_false[key] = {"time trigg": times[i], "time corr": time_corr, "diff": diff,
                                         "channels": channels}
                    i += 1

        return glitch_corr, len(glitch_false.keys()), glitch_false


def find_glitch_matches(ch1, ch2, ch3=None, ch4=None, ch5=None, ch6=None, check_times=False, input_glitches=None):
    """
    compare all given channels for glitch entries
    if check_times=True: just checking how many glitches we get from all given channels based off checking the time_corr
    of the input channels (which have already been checked for corresponding input times this is just to count how many
    we have if we tally them all up from multiple channels, doing this to be able to see which glitches are
    missed by EVERY channel)
    """

    channel_list = [ch1, ch2, ch3, ch4, ch5, ch6]

    all_seen_glitches = {}
    already_seen_glitch_times = []

    for i in range(len(channel_list)):
        ch = channel_list[i]
        if ch is None:
            pass
        else:
            for key, items in ch.items():
                time_corr = items['time corr']
                glitch_data = items
                if time_corr in all_seen_glitches.keys():

                    current_channels = all_seen_glitches[time_corr]['channels']
                    new_channels = glitch_data['channels']

                    for chan in new_channels:
                        all_seen_glitches[time_corr]['channels'].append(chan)

                    all_seen_glitches[time_corr]['channels'] = \
                        np.unique(np.array(all_seen_glitches[time_corr]['channels'])).tolist()

                else:
                    all_seen_glitches[time_corr] = {'time trigg': items['time trigg'], 'amp corr': items['amp corr'],
                                                    'freq': items['freq'], 'dur': items['dur'], 'snr': items['snr'],
                                                    'channels': items['channels']}
                    all_seen_glitches[time_corr]['channels'] = \
                        np.unique(np.array(all_seen_glitches[time_corr]['channels'])).tolist()

    return all_seen_glitches


def missing_glitches(found_glitches, real_glitches, amp_cutoff=None):
    """return the time, beta and level of glitches NOT picked up by the Q-transform"""
    missed_glitch, seen_glitch, k = {}, {}, 0
    seen_glitches = []
    if amp_cutoff is None:
        for key in found_glitches.keys():
            glitch_time = key
            glitch_data = found_glitches[key]
            if glitch_time in real_glitches['time']:
                idx = np.where(glitch_time == np.array(real_glitches['time']))[0]
                seen_glitch[k] = {'time trigg': glitch_data['time trigg'], 'time corr': glitch_time,
                                  'freq': glitch_data['freq'], 'beta': real_glitches['beta'][idx],
                                  'amplitude': real_glitches['amplitude'][idx],
                                  'channels': glitch_data['channels']}
                seen_glitches.append(glitch_time)
                k += 1
        for i in range(len(real_glitches['time'])):
            if real_glitches['time'][i] not in seen_glitches:
                missed_glitch[i] = {'time corr': real_glitches['time'][i], 'beta': real_glitches['beta'][i],
                                    'amplitude': real_glitches['amplitude'][i]}

    else:
        for key in found_glitches.keys():
            glitch_time = key
            glitch_data = found_glitches[key]
            if glitch_time in real_glitches['time']:
                idx = np.where(glitch_time == np.array(real_glitches['time']))[0]
                if real_glitches['amplitude'][idx] >= amp_cutoff:
                    seen_glitch[k] = {'time trigg': glitch_data['time trigg'], 'time corr': glitch_time,
                                      'freq': glitch_data['freq'], 'beta': real_glitches['beta'][idx],
                                      'amplitude': real_glitches['amplitude'][idx],
                                      'channels': glitch_data['channels']}
                    seen_glitches.append(glitch_time)
                    k += 1
        for i in range(len(real_glitches)):
            if real_glitches['time'][i] not in seen_glitches:
                if real_glitches['amplitude'][i] >= amp_cutoff:
                    missed_glitch[i] = {'time corr': real_glitches['time'][i], 'beta': real_glitches['beta'][i],
                                        'amplitude': real_glitches['amplitude'][i]}

    return missed_glitch, seen_glitch


def sort_glitches(glitches, which_glitches):
    """return times, betas, and levels"""

    times_trigg, times_real, betas, levs, freqs, channels = [], [], [], [], [], []
    if which_glitches == 'seen':
        for i in glitches.keys():
            times_trigg.append(glitches[i]['time trigg'])
            times_real.append(glitches[i]['time corr'])
            betas.append(glitches[i]['beta'])
            levs.append(glitches[i]['amplitude'])
            freqs.append(glitches[i]['freq'])
            channels.append(glitches[i]['channels'])
    elif which_glitches == 'false':
        for i in glitches.keys():
            times_trigg.append(glitches[i]['time trigg'])
            times_real.append(glitches[i]['time corr'])
            freqs.append(glitches[i]['freq'])
            channels.append(glitches[i]['channels'])
    else:  # missing
        for i in glitches.keys():
            times_real.append(glitches[i]['time corr'])
            betas.append(glitches[i]['beta'])
            levs.append(glitches[i]['amplitude'])

    return times_trigg, times_real, betas, levs, freqs, channels


def analysis(fname_list, white_dict, cutoff=None, need_amp=False, remove_fp=False, need_false=False, return_just=None,
             coincidence_win=None, inputs=None, h5_path=None, txt_path=None):

    obs_tdi = TimeSeriesDict.read(h5_path)
    obs_tdi["T"] = (obs_tdi["X"] + obs_tdi["Y"] + obs_tdi["Z"]) / np.sqrt(3.0)
    obs_tdi["E"] = (obs_tdi["X"] - 2.0 * obs_tdi["Y"] + obs_tdi["Z"]) / np.sqrt(6.0)
    obs_tdi["EV"] = (obs_tdi["Y"] - 2.0 * obs_tdi["Z"] + obs_tdi["X"]) / np.sqrt(6.0)

    # TODO: Fix all the paths here

    if inputs is None:
        input_glitches_fname = txt_path
        input_glitches_data = np.genfromtxt(input_glitches_fname)
        input_glitch_times = input_glitches_data[1:, 5]
        input_glitch_beta = input_glitches_data[1:, 7]
        input_glitch_amplitudes = input_glitches_data[1:, -1]
        input_glitch_all = {'time': input_glitch_times, 'level': input_glitch_amplitudes, 'beta': input_glitch_beta}
    else:
        input_glitch_all = inputs
        input_glitch_times = input_glitch_all['time']
        input_glitch_beta = input_glitch_all['beta']
        input_glitch_amplitudes = input_glitch_all['level']

    # Load trigger files
    X_cluster_trigger = np.genfromtxt(PATH_qtrans + fname_list[0])
    Y_cluster_trigger = np.genfromtxt(PATH_qtrans + fname_list[1])
    Z_cluster_trigger = np.genfromtxt(PATH_qtrans + fname_list[2])
    T_cluster_trigger = np.genfromtxt(PATH_qtrans + fname_list[3])
    E_cluster_trigger = np.genfromtxt(PATH_qtrans + fname_list[4])
    EV_cluster_trigger = np.genfromtxt(PATH_qtrans + fname_list[5])

    # Label artefacts as glitches or GW
    found_glitches, found_gw = label_glitch(X_cluster_trigger, Y_cluster_trigger, Z_cluster_trigger,
                                            tdiT=T_cluster_trigger, tdiE=E_cluster_trigger, tdiEV=EV_cluster_trigger,
                                            t_range=10.0)

    print("Finding Amplitudes of found glitches")

    found_amplitudes_xy = find_amplitudes('XY', found_glitches['XY'], found_glitches['XY dur'],
                                          found_glitches['XY freq'], found_glitches['XY snr'],
                                          obs_tdi['X'], obs_tdi['Y'], need_amp=need_amp)
    found_amplitudes_xz = find_amplitudes('XZ', found_glitches['XZ'], found_glitches['XZ dur'],
                                          found_glitches['XZ freq'], found_glitches['XZ snr']
                                          , obs_tdi['X'], obs_tdi['Z'], need_amp=need_amp)
    found_amplitudes_yz = find_amplitudes('YZ', found_glitches['YZ'], found_glitches['YZ dur'],
                                          found_glitches['YZ freq'], found_glitches['YZ snr'],
                                          obs_tdi['Y'], obs_tdi['Z'], need_amp=need_amp)
    found_amplitudes_t = find_amplitudes('T', found_glitches['T'], found_glitches['T dur'],
                                         found_glitches['T freq'], found_glitches['T snr'],
                                         obs_tdi['T'], None, need_amp=need_amp)

    found_amplitudes_e = find_amplitudes('E', found_glitches['E'], found_glitches['E dur'],
                                         found_glitches['E freq'], found_glitches['E snr'],
                                         obs_tdi['E'], None, need_amp=need_amp)
    found_amplitudes_ev = find_amplitudes('EV', found_glitches['EV'], found_glitches['EV dur'],
                                         found_glitches['EV freq'], found_glitches['EV snr'],
                                         obs_tdi['EV'], None, need_amp=need_amp)

    print("Remove False Positives (wippy)")

    ko_xy, ko_xz, ko_yz, ko_t, ko_e, ko_ev = 0, 0, 0, 0, 0, 0
    kicked_out_all = {'XY': [], 'XZ': [], 'YZ': [], 'T': []}
    if remove_fp:
        keep_glitches_all, kicked_out_all = \
            find_false_positives(found_amplitudes_e, found_amplitudes_ev, found_amplitudes_t, found_amplitudes_xy,
                                 found_amplitudes_xz, found_amplitudes_yz, white_dict)
        found_amplitudes_xy = keep_glitches_all['XY']
        found_amplitudes_xz = keep_glitches_all['XZ']
        found_amplitudes_yz = keep_glitches_all['YZ']
        found_amplitudes_t = keep_glitches_all['T']

    print('How many are real?')

    if coincidence_win is None:
        coincidence_win = 1000

    # Find Corresponding Glitches from the injected glitch file
    glitch_corr_xy, false_xy, glitch_false_xy = \
        find_corresponding_glitch(input_glitch_all, found_amplitudes_xy, coincidence_win, ['X', 'Y'])
    print("xy glitch found: ", len(glitch_corr_xy), "w/ false: ", false_xy,
          "kicked out", len(kicked_out_all['XY']), "all in", len(found_amplitudes_xy))

    glitch_corr_xz, false_xz, glitch_false_xz = \
        find_corresponding_glitch(input_glitch_all, found_amplitudes_xz, coincidence_win, ['X', 'Z'])
    print("xz glitch found: ", len(glitch_corr_xz), "w/ false: ", false_xz,
          "kicked out", len(kicked_out_all['XZ']), "all in", len(found_amplitudes_xz))

    glitch_corr_yz, false_yz, glitch_false_yz = \
        find_corresponding_glitch(input_glitch_all, found_amplitudes_yz, coincidence_win, ['Y', 'Z'])
    print("yz glitch found: ", len(glitch_corr_yz), "w/ false: ", false_yz,
          "kicked out", len(kicked_out_all['YZ']), "all in", len(found_amplitudes_yz))

    glitch_corr_t, false_t, glitch_false_t = \
        find_corresponding_glitch(input_glitch_all, found_amplitudes_t, coincidence_win, ['T'])
    print("t glitch found: ", len(glitch_corr_t), "w/ false: ", false_t,
          "kicked out", len(kicked_out_all['T']), "all in", len(found_amplitudes_t))

    glitch_corr_e, false_e, glitch_false_e = \
        find_corresponding_glitch(input_glitch_all, found_amplitudes_e, coincidence_win, ['E'])
    print("e glitch found: ", len(glitch_corr_e), "w/ false: ", false_e,
          "kicked out", ko_e, "all in", len(found_amplitudes_e))

    glitch_corr_ev, false_ev, glitch_false_ev = \
        find_corresponding_glitch(input_glitch_all, found_amplitudes_ev, coincidence_win, ['EV'])
    print("ev glitch found: ", len(glitch_corr_ev), "w/ false: ", false_ev,
          "kicked out", ko_ev, "all in", len(found_amplitudes_ev))

    if return_just is None:

        all_matched_glitches = \
            find_glitch_matches(glitch_corr_xy, glitch_corr_xz, glitch_corr_yz, glitch_corr_t,
                                check_times=True, input_glitches=input_glitch_times)

        missed_all, seen_all = missing_glitches(all_matched_glitches, input_glitch_all, amp_cutoff=None)

        missing_all_trigg, missing_all_times, missing_all_betas, missing_all_levels, missing_all_freqs, missing_all_ch = \
            sort_glitches(missed_all, which_glitches='missing')
        seen_all_trigg, seen_all_times, seen_all_betas, seen_all_levels, seen_all_freqs, seen_all_channels = \
            sort_glitches(seen_all, which_glitches='seen')

        seen_glitches = {'time trigg': seen_all_trigg, 'time corr': seen_all_times,
                         'beta': seen_all_betas, 'level': seen_all_levels, 'freq': seen_all_freqs,
                         'channels': seen_all_channels}
        missed_glitches = {'time trigg': missing_all_trigg, 'time corr': missing_all_times,
                           'beta': missing_all_betas, 'level': missing_all_levels, 'freq': missing_all_freqs,
                           'channels': missing_all_ch}

        if cutoff is not None:
            missed_all_c, seen_all_c = missing_glitches(all_matched_glitches, input_glitch_all, amp_cutoff=cutoff)
            missing_all_trigg_c, missing_all_times_c, missing_all_betas_c, \
                missing_all_levels_c, missing_all_freqs_c, missing_all_ch_c = \
                sort_glitches(missed_all_c, which_glitches='missing')
            seen_all_trigg_c, seen_all_times_c, seen_all_betas_c, \
                seen_all_levels_c, seen_all_freqs_c, seen_all_channels_c = \
                sort_glitches(seen_all_c, which_glitches='seen')

            seen_cut_glitches = {'time trigg': seen_all_trigg_c, 'time corr': seen_all_times_c,
                                 'beta': seen_all_betas_c, 'level': seen_all_levels_c, 'freq': seen_all_freqs_c,
                                 'channels': missing_all_ch_c}
            missed_cut_glitches = {'time trigg': missing_all_trigg_c, 'time corr': missing_all_times_c,
                                   'beta': missing_all_betas_c, 'level': missing_all_levels_c, 'freq': missing_all_freqs_c,
                                   'channels': seen_all_channels_c}

            if need_false:
                false_dict = {}
                false_list = [glitch_false_xy, glitch_false_xz, glitch_false_yz, glitch_false_t, glitch_false_e]

                chs = ['XY', 'XZ', 'YZ', 'T', 'E']
                for i in range(len(false_list)):
                    if len(chs[i]) > 1:
                        obs = [obs_tdi[chs[i][0]], obs_tdi[chs[i][1]]]
                    else:
                        obs = [obs_tdi[chs[i]]]
                    false_dict[chs[i]] = get_amplitude(false_list[i], obs)
                return seen_glitches, missed_glitches, seen_cut_glitches, missed_cut_glitches, false_dict

            return seen_glitches, missed_glitches, seen_cut_glitches, missed_cut_glitches

        else:

            if need_false:
                false_dict = {}
                false_list = [glitch_false_xy, glitch_false_xz, glitch_false_yz, glitch_false_t, glitch_false_e]
                chs = ['XY', 'XZ', 'YZ', 'T', 'E']
                for i in range(len(false_list)):
                    if len(chs[i]) > 1:
                        obs = [obs_tdi[chs[i][0]], obs_tdi[chs[i][1]]]
                    else:
                        obs = [obs_tdi[chs[i]]]
                    false_dict[chs[i]] = get_amplitude(false_list[i], obs)
                return seen_glitches, missed_glitches, false_dict

            return seen_glitches, missed_glitches

    else:
        info_dict = {'XY': {'triggers': found_amplitudes_xy, 'found glitches': glitch_corr_xy,
                            'false glitches': glitch_false_xy},
                     'XZ': {'triggers': found_amplitudes_xz, 'found glitches': glitch_corr_xz,
                            'false glitches': glitch_false_xz},
                     'YZ': {'triggers': found_amplitudes_yz, 'found glitches': glitch_corr_yz,
                            'false glitches': glitch_false_yz},
                     'T': {'triggers': found_amplitudes_t, 'found glitches': glitch_corr_t,
                           'false glitches': glitch_false_t},
                     'E': {'triggers': found_amplitudes_e, 'found glitches': glitch_corr_e,
                           'false glitches': glitch_false_e},
                     'EV': {'triggers': found_amplitudes_ev, 'found glitches': glitch_corr_ev,
                            'false glitches': glitch_false_ev}}
        info_dict_return = {}
        for ch in return_just:
            info_dict_return[ch] = info_dict[ch]

        if remove_fp:
            return info_dict_return
        else:
            return info_dict_return



