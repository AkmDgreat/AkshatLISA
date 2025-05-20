"""
main file to run both make_glitch and simulate glitch as one
they both however can be run separately
"""

import os
import make_glitch as mg
import ldc.io.yml as ymlio
import inject_glitch as ig
from ldc.utils.logging import init_logger


PATH_cd = os.getcwd()
PATH_lgs = os.path.abspath(os.path.join(PATH_cd, os.pardir))  # PATH to lisa_glitch_simulation directory
PATH_test = os.path.join(PATH_lgs, 'testing/')
PATH_io = os.path.join(PATH_lgs, 'input_output')
PATH_tdi_out = os.path.join(PATH_lgs, 'final_tdi_outputs')


def init_cl():

    import argparse
    parser = argparse.ArgumentParser()
    # inject_glitch arguments
    parser.add_argument('--path-input', type=str, default=PATH_io, help="Path to input glitch files")
    parser.add_argument('--path-output', type=str, default=PATH_tdi_out, help="Path to save output tdi files")
    parser.add_argument('--glitch-h5-mg-output', type=str, default="glitch", help="Glitch output h5 file")
    parser.add_argument('--glitch-txt-mg-output', type=str, default="glitch", help="Glitch output txt file")
    parser.add_argument('--tdi-output-file', type=str,
                        default="final_tdi", help="Glitch output h5 file for inject_glitch")

    # make_glitch arguments
    parser.add_argument('--config-input', type=str, default="pipeline_cfg", help="Pipeline config file")
    parser.add_argument('--glitch-config-input', type=str, default="glitch_cfg", help="Glitch config file")
    # parser.add_argument('--no-noise', type=bool, default=False, help="Whether or not to add noise")
    # parser.add_argument('--no-gaps', type=bool, default=True, help="Whether or not to add gaps")
    parser.add_argument('--testing', type=bool, default=False, help="Testing")
    parser.add_argument('--clean', type=bool, default=False, help="Clean data set")
    parser.add_argument('-l', '--log', default="", help="Log file")

    args = parser.parse_args()
    logger = init_logger(args.log, name='lisaglitch.glitch')

    return args


def prep_config(glitch_config, segments):

    cfg = ymlio.load_config(PATH_test + glitch_config)
    t_max = cfg["t_max"].to("s").value

    if t_max % segments == 0.0:
        ...  # TODO what even is this
    else:
        segments += 1


if __name__ == "__main__":

    main_args = init_cl()
    print('main_args', main_args)

    print('-- INTO make_glitch --')
    # create glitches
    glitch_file_h5, glitch_file_txt = mg.main(main_args)
    print('-- DONE make_glitch --')

    if main_args.clean:
        print('-- INTO inject_glitch --')
        ig.main(glitch_file_h5, glitch_file_txt, main_args.tdi_output_file, clean=main_args.clean)
        print('-- DONE inject_glitch --')
    else:
        # inject glitches
        print('-- INTO inject_glitch --')
        ig.main(glitch_file_h5, glitch_file_txt, main_args.tdi_output_file)
        print('-- DONE inject_glitch --')



