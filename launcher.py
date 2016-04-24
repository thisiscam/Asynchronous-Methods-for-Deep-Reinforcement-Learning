import ale_python_interface
import argparse

class Defaults:
    ROM = 'breakout.bin'
    REPEAT_ACTION_PROBABILITY = 0
    NPROCESS = 4
    FRAME_SKIP = 4
    RESIZE_METHOD = 'scale'
    MAX_START_NULLOPS = 30
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE='true'
    UPDATE_DISP = 'true'
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95 # (Rho)
    RMS_EPSILON = .01
    MOMENTUM = 0.0 # Note that the "momentum" value mentioned in the Nature
                 # paper is not used in the same way as a traditional momentum
                 # term.  It is used to track gradient for the purpose of
                 # estimating the standard deviation. This package uses
                 # rho/RMS_DECAY to track both the history of the gradient
                 # and the squared gradient.

def load_ale(parameters, rng, update_disp=False):
    ale = ale_python_interface.ALEInterface()
    ale.setInt('random_seed', rng.randint(1000))
    if update_disp:
        import sys
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            ale.setBool('sound', False) # Sound doesn't work on OSX

    ale.setBool('display_screen', update_disp)
    ale.setFloat('repeat_action_probability',
                 parameters.repeat_action_probability)
    ale.loadROM(parameters.rom)
    return ale

def process_args(args, defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rom', dest="rom", default=defaults.ROM,
                        help='ROM to run (default: %(default)s)')
    parser.add_argument('--repeat-action-probability',
                        dest="repeat_action_probability",
                        default=defaults.REPEAT_ACTION_PROBABILITY, 
                        type=float,
                        help=('Probability that action choice will be ' +
                              'ignored (default: %(default)s)'))
    parser.add_argument('-np', "--nprocesses", dest="np", 
                        type=int,
                        default=defaults.NPROCESS,
                        help="number of processes for async run")
    parser.add_argument('--frame-skip', dest="frame_skip",
                        default=defaults.FRAME_SKIP, type=int,
                        help='Every how many frames to process '
                        '(default: %(default)s)')
    parser.add_argument('--resize-method', dest="resize_method",
                        type=str, default=defaults.RESIZE_METHOD,
                        help=('crop|scale (default: %(default)s)'))
    parser.add_argument('--max-start-nullops', dest="max_start_nullops",
                        type=int, default=defaults.MAX_START_NULLOPS,
                        help=('Maximum number of null-ops at the start ' +
                              'of games. (default: %(default)s)'))
    parser.add_argument('--death-ends-episode', dest="death_ends_episode",
                        type=str, default=defaults.DEATH_ENDS_EPISODE,
                        help=('true|false (default: %(default)s)'))
    parser.add_argument('--update_disp', dest="update_disp",
                        type=str, default=defaults.UPDATE_DISP,
                        help=('true|false (default: %(default)s)'))
    parser.add_argument('--learning-rate', dest="learning_rate",
                        type=float, default=defaults.LEARNING_RATE,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--rms-decay', dest="rms_decay",
                        type=float, default=defaults.RMS_DECAY,
                        help='Decay rate for rms_prop (default: %(default)s)')
    parser.add_argument('--rms-epsilon', dest="rms_epsilon",
                        type=float, default=defaults.RMS_EPSILON,
                        help='Denominator epsilson for rms_prop ' +
                        '(default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=defaults.MOMENTUM,
                        help=('Momentum term for Nesterov momentum. '+
                              '(default: %(default)s)'))
    parser.add_argument('--discount', type=float, default=defaults.DISCOUNT,
                        help='Discount rate')
    parameters = parser.parse_args(args)

    if parameters.death_ends_episode == 'true':
        parameters.death_ends_episode = True
    elif parameters.death_ends_episode == 'false':
        parameters.death_ends_episode = False
    else:
        raise ValueError("--death-ends-episode must be true or false")

    if parameters.update_disp == 'true':
        parameters.update_disp = True
    elif parameters.update_disp == 'false':
        parameters.update_disp = False
    else:
        raise ValueError("--update_disp must be true or false")

    return parameters
