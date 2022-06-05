from CoachMutliProc import Coach
from antichess3.pytorch.NNet import NNetWrapper as nn
from antichess3.AntiChessGame import AntiChessGame as Game

from utils_multi_proc import *
import os

"""
Before using multiprocessing, please check 2 things before use this script.
1. The number of PlayPool should not over your CPU's core number.
2. Make sure all Neural Network which each process created can store in VRAM at same time. Check your NN size before use this.
"""
args = dotdict({
    'numIters': 2100,
    'numEps': 100,
    'tempThreshold': 40,
    'updateThreshold': 0.57,
    'maxlenOfQueue': 200000,
    'dirichletAlpha': 0.1, # α = {0.3, 0.15, 0.03} for chess, shogi and Go respectively, scaled in inverse proportion to the approximate number of legal moves in a typical position
    'numMCTSSims': 75,
    'cpuct': 2.0,
    'multiGPU': False,
    'setGPU': '0',
    # The total number of games when self-playing is:
    # Total = numSelfPlayProcess * numPerProcessSelfPlay
    'numSelfPlayProcess': 5,
    'numPerProcessSelfPlay':20,
    # The total number of games when against-playing is:
    # Total = numAgainstPlayProcess * numPerProcessAgainst
    'numAgainstPlayProcess': 5,
    'numPerProcessAgainst': 8,
    'checkpoint': './temp/antichess/final/torch/numIters=200numEPS=99tempThre=40da=0.1MCTSSims=75cpuct=2.0openingBook=YES',
    'numItersForTrainExamplesHistory': 15,
})

if __name__=="__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    g = Game(8)
    c = Coach(g, args)
    c.learn()