from make_glitch import make_glitch
from inject_glitch import inject_glitch
import argparse

"""
args=(
    --glitch_cfg_input      glitch_exact.yml
    --glitch_output_h5      glitch_exact.h5
    --glitch_output_txt     glitch_exact.txt
    --tdi_output_h5         glitch_exact.h5
    --simulation_output_h5  glitch_exact.h5
    --pipe_cfg_input        pipeline_cfg.yml
    --orbit_input_h5        orbits.h5
)

python main.py "${args[@]}"
"""

def init_cl():
    """Initialize commandline arguments and return Namespace object with all
    given commandline arguments.
    """

    parser = argparse.ArgumentParser()

    # FILE MANAGEMENT
    parser.add_argument(
        "--pipe_cfg_input",
        type=str,
        default="",
        help="Pipeline config file name"
    )
    parser.add_argument(
        "--glitch_cfg_input",
        type=str,
        default="",
        help="Glitch config file name",
    )
    parser.add_argument(
        "--orbit_input_h5",
        type=str,
        default="orbits.h5",
        help="Orbit .h5 file name",
    )
    parser.add_argument(
        "--glitch_output_h5",
        type=str,
        default="default_glitch_output.h5",
        help="Glitch output .h5 file name",
    )
    parser.add_argument(
        "--glitch_output_txt",
        type=str,
        default="default_glitch_output.txt",
        help="Glitch output .txt file name",
    )
    parser.add_argument(
        "--tdi_output_h5",
        type=str,
        default="default_tdi_output.h5",
        help="TDI output .h5 file name",
    )
    parser.add_argument(
        "--simulation_output_h5",
        type=str,
        default="default_simulation_output.h5",
        help="LISA simulation output .h5 file name",
    )

    # LISA INSTRUMENT ARGUMENTS
    parser.add_argument(
        "--disable_noise",
        type=bool,
        default=True,
        help="Simulate LISA instruments without noise?"
    )

    # SEED
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to ensure deterministic outputs"
    )

    return parser.parse_args()


def main():
    args = init_cl()

    make_glitch(args)

    inject_glitch(
        args.orbit_input_h5,
        args.glitch_output_h5,
        args.glitch_output_txt,
        args.simulation_output_h5,
        args.tdi_output_h5,
        args.disable_noise,
    )


if __name__ == "__main__":
    main()