from ArenaMultiProc import Arena
from MCTS import MCTS
import time
from antichess3.AntiChessGame import AntiChessGame
from antichess3.AntiChessPlayers import *
from antichess3.pytorch.NNet import NNetWrapper as NNet
import os
import numpy as np
import tensorflow as tf
import multiprocessing
from utils_multi_proc import *
from antichess3.progress.bar import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

def Async_Play(game,args,iter_num,bar):
    bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(i=iter_num+1,x=args.numPlayGames,total=bar.elapsed_td, eta=bar.eta_td)
    bar.next()
    # set gpu
    if(args.multiGPU):
        if(iter_num%2==0):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    # set gpu growth
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)

    # create NN
    model1 = NNet(game)

    # try load weight
    try:
        model1.load_checkpoint(folder=args.model1Folder, filename=args.model1FileName)
    except:
        print("load model1 fail")
        pass


    # create MCTS
    mcts1 = MCTS(game, model1, args)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    rp1 = RandomPlayer(game).play
    #stockfish1 = StockFishPlayer(game, 0.001).play

    # each process play 2 games
    arena = Arena(rp1,rp1, game)
    arena.displayBar = False
    oneWon,twoWon, draws = arena.playGames(10)
    return oneWon,twoWon, draws

if __name__=="__main__":
    """
    Before using multiprocessing, please check 2 things before use this script.
    1. The number of PlayPool should not over your CPU's core number.
    2. Make sure all Neural Network which each process created can store in VRAM at same time. Check your NN size before use this.
    """
    args = dotdict({
    'numMCTSSims': 20,
    'cpuct': 1.00,

    'multiGPU': False,  # multiGPU only support 2 GPUs.
    'setGPU': '0',
    'numPlayGames': 20,  # total num should x2, because each process play 2 games.
    'numPlayPool': 4,   # num of processes pool.

    'model1Folder': './temp/antichess/reduced_moves/torch/QROOKSKINGPAWNSFRONTx200x100x50/',
    'model1FileName': 'best.pth.tar',


    })

    def ParallelPlay(g):
        bar = Bar('Play', max=args.numPlayGames)
        pool = multiprocessing.Pool(processes=args.numPlayPool)
        res = []
        result = []
        for i in range(args.numPlayGames):
            res.append(pool.apply_async(Async_Play,args=(g,args,i,bar)))
        pool.close()
        print("Join")
        pool.join()

        oneWon = 0
        twoWon = 0
        draws = 0
        for i in res:
            result.append(i.get())
        for i in result:
            oneWon += i[0]
            twoWon += i[1]
            draws += i[2]
        print("Model 1 Win:",oneWon," Model 2 Win:",twoWon," Draw:",draws)


    g = AntiChessGame(8)

    # parallel version
    print("here")
    start = time.time()
    ParallelPlay(g)
    stop = time.time()
    print(stop-start)
