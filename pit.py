import Arena
from MCTS import MCTS

from antichess3.AntiChessGame import  AntiChessGame
from antichess3.AntiChessPlayers import *


from antichess3.pytorch.NNet import NNetWrapper as NNet

from othello.pytorch.NNet import NNetWrapper as NNet
from othello.OthelloGame import  OthelloGame
from othello.OthelloPlayers import *

import numpy as np
from utils import *

import time

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

#g = AntiChessGame(8)
#g = TicTacToeGame()
#g = ChessGame()
g = OthelloGame(6)

# all players
rp1 = RandomPlayer(g).play
rp2 = RandomPlayer(g).play

#stockfish1 = StockFishPlayer(g,2500).play
#stockfish2 = StockFishPlayer(g,500).play


#hp = HumanPawnsPlayer(g).play

# nnet players
#n1 = NNet(g)
#n1.load_checkpoint('./temp/antichess/final/torch/100x50x20x40x1/','best.pth.tar')

#args1 = dotdict({'numMCTSSims': 20, 'cpuct':1.0})
#mcts1 = MCTS(g, n1, args1)
#n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

#n2 = NNet(g)
#n2.load_checkpoint('./temp/antichess/final/torch/100x50x20x40x1/','checkpoint_93.pth.tar')

#args2 = dotdict({'numMCTSSims': 20, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))


arena = Arena.Arena(rp1,rp2, g)


start = time.time()
print(arena.playGames(40,verbose=False))
end = time.time()
print(end-start)

