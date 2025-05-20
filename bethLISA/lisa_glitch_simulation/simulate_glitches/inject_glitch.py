"""
Inject glitches into LISA data
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from pytdi.michelson import X2, Y2, Z2
from ldc.utils.logging import init_logger, close_logger
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from lisainstrument.containers import ForEachMOSA
from lisainstrument import Instrument
from pytdi import Data

start_time = time.time()  

PATH_cd = os.getcwd()
PATH_lgs = os.path.abspath(os.path.join(PATH_cd, os.pardir))  # PATH to lisa_glitch_simulation directory
PATH_io = os.path.join(os.path.abspath(os.path.join(PATH_cd, os.pardir)), 'glitch_txt_and_h5_files')
PATH_tdi_out = os.path.join(PATH_lgs, 'tdi_outputs')

TDI_VAR = [X2, Y2, Z2]
TDI_NAMES = ['X', 'Y', 'Z']

"""
args=(
    --glitch-h5-mg-output     glitch.h5 
    --glitch-txt-mg-output    glitch.txt 
    --tdi-output-file         final_tdi.h5 
    --glitches true
    --noise true
)

python inject_glitch.py "${args[@]}"
"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def init_cl():
    """
    initialize the command line arguments
    """

    import argparse
    parser = argparse.ArgumentParser()
    # Main arguments
    parser.add_argument(
        '--path-input', 
        type=str, 
        default=PATH_io, 
        help="Path to input glitch files"
    )
    parser.add_argument(
        '--path-output', 
        type=str, 
        default=PATH_cd, 
        help="Path to save output tdi files"
    )
    parser.add_argument('--glitch-h5-mg-output', type=str, default="glitch.h5", help="Glitch output h5 file")
    parser.add_argument('--glitch-txt-mg-output', type=str, default="glitch.txt", help="Glitch output txt file")
    parser.add_argument('--tdi-output-file', type=str, default="final_tdi", help="tdi output h5 file")

    parser.add_argument('--glitches', type=str2bool, default=False, help="Want Glitches?")

    parser.add_argument('--noise', type=bool, default=True, help="Want noise?")
    parser.add_argument('-l', '--log', default="", help="Log file")
    args = parser.parse_args()
    logger = init_logger(args.log, name='lisaglitch.glitch')

    return args


def init_inputs(glitch_info, old_file=False):
    """
    Initialize input variables from a glitch info file.

    Args:
        glitch_info (str): Filename of the glitch info CSV/TXT.
        old_file (bool): Whether file uses the old format (shifts t0 index).

    Returns:
        dict: Parsed inputs including sample count, timing, upsampling, and noise specs.
    """
    # Load glitch metadata
    path = os.path.join(PATH_io, glitch_info)
    g_info = np.genfromtxt(path)

    # Base parameters from second row
    row = g_info[1]
    n_samples = int(row[1])
    g_dt = float(row[2])

    # t0 and upsampling differ by format
    if old_file:
        g_t0 = float(row[3])
        g_physics_upsampling = 1.0
    else:
        g_t0 = float(row[4])
        g_physics_upsampling = float(row[3])

    # Derived physical timestep
    dt_physic = g_dt / g_physics_upsampling

    # Fixed instrument parameters
    central_freq = 2.816e14
    aafilter = None
    noise_dict = {
        "backlinknoise": 3e-12,
        "accnoise":      2.4e-15,
        "readoutnoise":  6.35e-12,
    }

    return {
        'n_samples': n_samples,
        'dt': g_dt,
        't0': g_t0,
        'physics_upsampling': g_physics_upsampling,
        'dt_physic': dt_physic,
        'central_freq': central_freq,
        'aafilter': aafilter,
        'noise_dict': noise_dict,
    }


def simulate_lisa(glitch_file, glitch_inputs, noise, glitches):
    """Simulate the LISA instrument, optionally injecting glitches and noise."""
    # Extract common parameters
    noise_dict = glitch_inputs['noise_dict']
    common_kwargs = {
        'physics_upsampling': glitch_inputs['physics_upsampling'],
        'aafilter': glitch_inputs['aafilter'],
        'size': glitch_inputs['n_samples'],
        'dt': glitch_inputs['dt'],
        'central_freq': glitch_inputs['central_freq'],
        'backlink_asds': noise_dict['backlinknoise'],
        'testmass_asds': noise_dict['accnoise'],
    }
    print(common_kwargs)

    if glitches:
        common_kwargs['glitches'] = glitch_file
        print("Glitch injected")

    print(common_kwargs)

    # Initialize instrument
    instrument = Instrument(**common_kwargs)

    # Noise configuration
    if noise:
        instrument.oms_isc_carrier_asds = ForEachMOSA(noise_dict['readoutnoise'])
        instrument.laser_asds = ForEachMOSA(0)  # Remove laser noise for PyTDI compatibility

        if not glitches:
            instrument.disable_clock_noises()
            instrument.modulation_asds = ForEachMOSA(0)
            instrument.disable_ranging_noises()
            instrument.disable_dopplers()
    else:
        instrument.disable_all_noises()

    # Run the simulation and return
    instrument.simulate()
    return instrument


def tdi_channels(i, channels, inputs, tdi_names):
    """create the TDI channels X, Y, Z using PyTDI

    Args
    i (lisainstrument simulation object): the simulation of a lisa-like set-up
    channels (PyTDI michelson variables): second gen michelson variables from PyTDI
    inputs (dict): dictionary of inputs from the glitch .txt file from make_glitch
    tdi_names (list): list of the TDI channel names in same order as channels

    Returns
    dict of all constructed TDI channels
    """

    tdis = TimeSeriesDict()
    for j in range(len(channels)):
        ch = channels[j]

        data = Data.from_instrument(i)
        data.delay_derivative = None

        built = ch.build(delays=data.delays, fs=data.fs)

        tdi_data = built(data.measurements)/inputs['central_freq']

        # Window out the tdi channels - tukey window
        win = tukey(tdi_data.size, alpha=0.001)
        tdis[tdi_names[j]] = TimeSeries(tdi_data*win, t0=inputs['t0'], dt=inputs['dt'])

    return tdis


def plot_tdi(tdi, tdi_name, xlims=None, ylims=None):

    times_arr = np.arange(0, 172800, 0.25)

    plt.figure(figsize=(10, 8))
    plt.plot(times_arr, tdi, label=f'raw TDI {tdi_name}')

    plt.title(f'TDI {tdi_name}')
    plt.xlabel('times [s]')
    plt.xlabel('amplitude')

    plt.legend()
    plt.show()

def save_tdi(tdi, output_fname, output_path):
    tdi.write(f'{output_path}/{output_fname}', overwrite=True, format='hdf5')

def main(args):
    tdi_start_t = time.time()

    inputs = init_inputs(args.glitch_txt_mg_output, old_file=True)
    sim = simulate_lisa(PATH_io + '/' + args.glitch_h5_mg_output, inputs, args.noise, args.glitches)
    tdi_dict = tdi_channels(sim, TDI_VAR, inputs, TDI_NAMES)
    save_tdi(tdi_dict, args.tdi_output_file, PATH_tdi_out)

    tdi_end_t = time.time()
    print("TDI Time: ")
    print("--- %s seconds ---" % (tdi_end_t - tdi_start_t))

"""Uncomment to run inject_glitch alone"""
if __name__ == "__main__":
    args = init_cl()
    print('main_args', args)
    main(args)