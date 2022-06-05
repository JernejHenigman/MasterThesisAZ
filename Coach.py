import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
from datetime import datetime

log = logging.getLogger(__name__)
from antichess3.Digits import *
import pickle
import chess.pgn
import time


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        # self.mcts = MCTS(self.game, self.nnet, self.args)
        self.mcts = MCTS(self.game, self.nnet, self.args, dirichlet_noise=True)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        self.time_mcts = 0
        self.time_nnet = 0
        self.time_nmcts = 0
        self.time_pmcts = 0
        self.time_play = 0

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:

            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)

            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)

            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                filehandler = open("returnList_normal", "wb+")
                pickle.dump([(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples], filehandler)
                filehandler.close()
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    # self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    self.mcts = MCTS(self.game, self.nnet, self.args, dirichlet_noise=True)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            print("Ns:", len(self.mcts.Ns))
            print("Nsa:", len(self.mcts.Nsa))
            print("Ps:", len(self.mcts.Ps))
            print("Qsa:", len(self.mcts.Qsa))
            print("Es:", len(self.mcts.Es))
            print("Vs:", len(self.mcts.Vs))

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='checkpoint_' + str(i) + '.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args, dirichlet_noise=True)

            start = time.time()
            self.nnet.train(trainExamples)
            end = time.time()
            self.time_nnet += (end - start)
            nmcts = MCTS(self.game, self.nnet, self.args, dirichlet_noise=True)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)

            start = time.time()
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            end = time.time()
            self.time_play += (end - start)

            self.time_mcts = self.mcts.time_sims + self.mcts.time_vali + self.mcts.time_next + self.mcts.time_pick + self.mcts.time_diri
            self.time_nmcts = nmcts.time_sims + nmcts.time_vali + nmcts.time_next + nmcts.time_pick + nmcts.time_diri
            self.time_pmcts = pmcts.time_sims + pmcts.time_vali + pmcts.time_next + pmcts.time_pick + pmcts.time_diri

            print("nmcts_time:", self.time_nmcts)
            print("pmcts_time:", self.time_pmcts)
            print("trai_time:", self.time_nnet)
            print("mcts_time:", self.time_mcts)
            print("play_time:", self.time_play)

            print("mdiri_time:", self.mcts.time_diri)
            print("msims_time:", self.mcts.time_sims)
            print("mvali_time:", self.mcts.time_vali)
            print("mnext_time:", self.mcts.time_next)
            print("mpick_time:", self.mcts.time_pick)
            print("mnnet_time:", self.mcts.time_nnet)

            print("ndiri_time:", nmcts.time_diri)
            print("nsims_time:", nmcts.time_sims)
            print("nvali_time:", nmcts.time_vali)
            print("nnext_time:", nmcts.time_next)
            print("npick_time:", nmcts.time_pick)
            print("nnnet_time:", nmcts.time_nnet)

            print("pdiri_time:", pmcts.time_diri)
            print("psims_time:", pmcts.time_sims)
            print("pvali_time:", pmcts.time_vali)
            print("pnext_time:", pmcts.time_next)
            print("ppick_time:", pmcts.time_pick)
            print("pnnet_time:", pmcts.time_nnet)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            print("examples:", sum([len(hist) for hist in self.trainExamplesHistory]))

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True