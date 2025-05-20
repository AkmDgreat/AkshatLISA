"""
support functions for make_glitch
"""
import os

PATH_cd = os.getcwd()
PATH_test = os.path.join(os.path.abspath(os.path.join(PATH_cd, os.pardir)), 'testing/')


def check_args(args):
    print('args', args)

    print('b4', type(args.path_input))
    if args.path_input is None:
        print('here')
        args.path_input = PATH_test
    print('af', args.path_input)
    quit()
