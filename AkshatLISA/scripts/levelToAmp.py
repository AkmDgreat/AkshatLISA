#!/usr/bin/env python
"""
scan_levels.py – empirically map glitch LEVEL → observed AMPLITUDE

Usage example
-------------
python scan_levels.py \
    --cfg glitch_exact.yml \
    --pipe_cfg pipeline_cfg.yml \
    --orbit orbits.h5 \
    --min 1e-8 --max 1e-10 --n 1000 \
    --left 5 --right 5 \
    --out levels_vs_amp.npz
"""
import argparse, subprocess, tempfile, shutil, os, yaml, h5py, numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# helpers                                                                    #
# --------------------------------------------------------------------------- #
def run_sim(cfg_path: str,
            pipe_cfg: str,
            orbit_h5: str,
            tag: str) -> str:
    """
    Launch `main.py` for *one* level value and return path to TDI output h5.
    Each run is sandboxed inside a temporary directory that is cleaned up
    automatically, so it never clobbers your main repo.
    """
    tmp = tempfile.mkdtemp(prefix=f"lvl_{tag}_")
    tdi_out = os.path.join(tmp, f"glitch_{tag}.h5")
    cmd = [
        "python", "main.py",
        "--glitch_cfg_input",  cfg_path,
        "--glitch_output_h5",  tdi_out,
        "--glitch_output_txt", os.path.join(tmp, f"glitch_{tag}.txt"),
        "--tdi_output_h5",     tdi_out,
        "--simulation_output_h5", tdi_out,
        "--pipe_cfg_input",    pipe_cfg,
        "--orbit_input_h5",    orbit_h5,
        "--disable_noise",     "False"
    ]
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tdi_out, tmp  # caller may decide when to delete


def find_peak_amp(h5_path: str,
                  t_inj: float,
                  left: int,
                  right: int):
    """Return the maximum |amplitude| in [idx-left, idx+right] around injection."""
    with h5py.File(h5_path, "r") as f:
        # assumes dataset 'X' – adjust if your key is different
        data = f["X"][:]
        t    = f["t"][:] if "t" in f else np.arange(len(data))
    idx = np.argmin(np.abs(t - t_inj))
    sli = slice(max(idx - left, 0), min(idx + right + 1, len(data)))
    return np.max(np.abs(data[sli]))


def logspace(min_v, max_v, n):
    # include both endpoints
    return np.logspace(np.log10(min_v), np.log10(max_v), num=n)


# --------------------------------------------------------------------------- #
# main driver                                                                 #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",  default="glitch_exact.yml")
    p.add_argument("--pipe_cfg",  default="pipeline_cfg.yml")
    p.add_argument("--orbit", default="orbits.h5")
    p.add_argument("--min",  type=float, default=1e-8)
    p.add_argument("--max",  type=float, default=1e-10)
    p.add_argument("--n",    type=int,   default=1000)
    p.add_argument("--left",  type=int,  default=5,
                   help="samples left of inj idx to search for peak")
    p.add_argument("--right", type=int,  default=5,
                   help="samples right of inj idx to search for peak")
    p.add_argument("--out",  default="levels_vs_amp.npz")
    args = p.parse_args()

    # read the *base* yaml once (everything but 'level' stays frozen)
    with open(args.cfg) as fh:
        base_cfg = yaml.safe_load(fh)

    t_inj = base_cfg["t_inj"][0]         # seconds
    levels = logspace(args.min, args.max, args.n)
    amps   = np.empty_like(levels)

    for k, lvl in enumerate(levels):
        cfg = dict(base_cfg)
        cfg["level"] = [float(lvl)]      # maintain list form
        tmp_cfg_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".yml", prefix="glitch_lvl_")
        with open(tmp_cfg_file.name, "w") as fh:
            yaml.safe_dump(cfg, fh)

        tag = f"{lvl:.1e}".replace("-", "m")  # safe filename chunk
        tdi_out, tmpdir = run_sim(tmp_cfg_file.name,
                                  args.pipe_cfg,
                                  args.orbit,
                                  tag)

        amps[k] = find_peak_amp(tdi_out, t_inj, args.left, args.right)

        # clean up heavy artefacts (comment out if you want to inspect files)
        shutil.rmtree(tmpdir, ignore_errors=True)
        os.unlink(tmp_cfg_file.name)

        print(f"[{k+1:>4}/{args.n}] level={lvl:.3e} → amp={amps[k]:.3e}")

    # save raw arrays for downstream use
    np.savez_compressed(args.out, level=levels, amp=amps)
    print(f"Saved data → {args.out}")

    # simple power-law fit in log-log
    a, b = np.polyfit(np.log10(levels), np.log10(amps), 1)  # amp ≈ 10^b · level^a
    print(f"Best log-log fit:  amp ≈ {10**b:.3e} · level^{a:.3f}")

    # plot
    plt.figure()
    plt.loglog(levels, amps, ".", ms=4, label="simulation")
    plt.loglog(levels, 10**b * levels**a, "-", lw=1,
               label=f"fit: amp ≈ {10**b:.2e}·level$^{a:.2f}$")
    plt.xlabel("glitch 'level' parameter")
    plt.ylabel("|amplitude| around $t_{inj}$")
    plt.title("Empirical mapping level → amplitude")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


"""
python ../../../AkshatLISA/scripts/levelToAmp.py \
  --cfg glitch_config/glitch_exact.yml \
  --pipe_cfg pipeline_config/pipeline_cfg.yml \
  --orbit orbit_data/orbits.h5
"""

