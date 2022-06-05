import logging

import coloredlogs

from Coach import Coach
from antichess3.AntiChessGame import AntiChessGame as Game
from antichess3.pytorch.NNet import NNetWrapper as nn
from utils import *
from antichess3.Digits import *
import numpy as np
#from othello.pytorch.NNet import NNetWrapper as nn
#from othello.OthelloGame import  OthelloGame
#from othello.OthelloPlayers import *


log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1,
    'numEps': 2,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 25,        #
    'updateThreshold': 0.60,    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 20,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.0,
    'dirichletAlpha': 0.1, # α = {0.3, 0.15, 0.03} for chess, shogi and Go respectively, scaled in inverse proportion to the approximate number of legal moves in a typical position

    'checkpoint': './temp/antichess/final/torch/100x50x20x40x1/',
    'load_model': False,
    'load_folder_file': ('./temp/antichess/final/torch/100x50x20x40x1/','checkpoint_01.pth.tar'),
     #'checkpoint': './temp/othello/final/torch/100x50x20x40x1/',
     #'load_model': False,
     #'load_folder_file': ('./temp/othello/final/torch/100x50x20x40x1/','checkpoint_76.pth.tar'),
    'numItersForTrainExamplesHistory': 40,

})


def get_move_from_action(action):
    if action in REVERSE_PROMOTIONS:
        move = REVERSE_PROMOTIONS[action]

    else:
        move = int2base(action, 8, 4)
        move = FILE_MAP_REVERSE[move[0]] + "" + RANK_MAP_REVERSE[move[1]] + "" + FILE_MAP_REVERSE[move[2]] + "" + \
               RANK_MAP_REVERSE[move[3]]

    return move


def get_action_from_move(move):
    x1, y1, x2, y2 = FILE_MAP[move[0]], RANK_MAP[move[1]], FILE_MAP[move[2]], RANK_MAP[move[3]]
    action = x1 + y1 * 8 + x2 * 8 * 2 + y2 * 8 * 3

    return action

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(8)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0],args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()




if __name__ == "__main__":
    main()
