from gwpy.timeseries import TimeSeriesDict

def readTdi(tdi_file_path):
    obs = TimeSeriesDict.read(tdi_file_path)
    data = obs["X"].value
    dt  = obs["X"].dt.value
    fs  = 1.0 / dt

    return data, fs