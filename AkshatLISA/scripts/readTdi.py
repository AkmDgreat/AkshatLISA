from gwpy.timeseries import TimeSeriesDict
import os as os

def readTdi(tdi_file_path):
    obs = TimeSeriesDict.read(tdi_file_path)
    data = obs["X"].value
    dt  = obs["X"].dt.value
    t = obs["X"].times.value 
    fs  = 1.0 / dt

    return data, fs, t

def getTdiPath(tdi_fileName):
    bethLISA_directory = os.path.join(os.getcwd(), "..", "..", "bethLISA")
    tdi_path = os.path.join(bethLISA_directory, 'lisa_glitch_simulation',
                        'tdi_outputs', tdi_fileName)
    
    return tdi_path