import os
from gwpy.timeseries import TimeSeriesDict
from scripts.psdAndPlot import psd_and_plot_hor, psd_and_plot

def psdAndPlotTDI(tdi_file_path, nper=4096, title="TDI-X", custom_wosa=False):
    """
    Load a TDI channel and run PSD+plot

    Parameters
    ----------
    tdi_file_path : str
        Name of the .h5 file in PATH_lgs/final_tdi_outputs/
    nper : int, optional
        Segment length for Welch PSD (default 4096).
    title : str, optional
        Plot title and also picks channel letter after the dash,
        e.g. "TDI-X" -> channel "X".
    """

    # read all channels
    obs = TimeSeriesDict.read(tdi_file_path)

    # infer channel key from title (last char after dash)
    channel = title.split('-')[-1]

    # extract arrays
    t   = obs[channel].times.value
    data = obs[channel].value
    dt  = obs[channel].dt.value

    # call your existing function
    psd_and_plot_hor(data, t, dt, nper=nper, title=title, custom_wosa=custom_wosa)