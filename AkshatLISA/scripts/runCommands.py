# pythonScript.py
import subprocess
import pathlib

def run_make_and_inject(
    directory: str = "lisa_glitch_simulation/simulate_glitches",
    h5_out: str = "day_glitch.h5",
    txt_out: str = "day_glitch.txt",
    tdi_make: str = "mg_tdi.h5",
    cfg_pipeline: str = "pipeline_cfg.yml",
    cfg_glitch: str = "glitch_cfg_day.yml",
    tdi_final: str = "day_glitch_tdi.h5",
):
    
    # 1) make glitches
    make_glitch_cmd = [
        "python", "make_glitch.py",
        "--glitch-h5-mg-output", h5_out,
        "--glitch-txt-mg-output", txt_out,
        "--tdi-output-file",      tdi_make,
        "--config-input",         cfg_pipeline,
        "--glitch-config-input",  cfg_glitch,
    ]
    subprocess.run(make_glitch_cmd, cwd=directory, check=True)

    # 2) inject glitches
    inject_glitch_cmd = [
        "python", "inject_glitch.py",
        "--glitch-h5-mg-output", h5_out,
        "--glitch-txt-mg-output", txt_out,
        "--tdi-output-file",      tdi_final,
        "--glitches", "true",
        "--noise", "true",
    ]
    subprocess.run(inject_glitch_cmd, cwd=directory, check=True)
