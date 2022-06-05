import logging
import math
import time
from antichess3.Digits import *
import numpy as np
import chess
EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, dirichlet_noise=False):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.dirichlet_noise = dirichlet_noise
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.time_sims = 0

        self.time_nnet = 0
        self.time_vali = 0
        self.time_next = 0
        self.time_diri = 0
        self.time_pick = 0

    def opening_book(self, board, book):
        total_moves = sum([move[2] for move in book])
        probs = np.zeros(4096)
        pre_moves = []
        for move in book:
            prob = move[2] / total_moves
            pre_moves.append((get_action_from_move(str(board.parse_san(move[0]))), prob))

        for pre_move in pre_moves:
            probs[pre_move[0]] = pre_move[1]
        return probs

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        probs = np.zeros(4096)
        if canonicalBoard.board.fullmove_number < 2:
            try:
                if canonicalBoard.board.fullmove_number == 1 and canonicalBoard.board.turn:
                    probs = self.opening_book(canonicalBoard.board, book1)

                elif canonicalBoard.board.fullmove_number == 1 and not canonicalBoard.board.turn:
                    copy_board = canonicalBoard.board.copy()
                    previous = copy_board.pop()
                    move_chosen_previously = copy_board.san(chess.Move.from_uci(str(previous)))
                    probs = self.opening_book(canonicalBoard.board, book2[move_chosen_previously])


                elif canonicalBoard.board.fullmove_number == 2 and canonicalBoard.board.turn:
                    copy_board = canonicalBoard.board.copy()
                    previous = copy_board.pop()
                    move_chosen_previously = copy_board.san(chess.Move.from_uci(str(previous)))
                    previous_previous = copy_board.pop()
                    move_chosen_previously_previously = copy_board.san(chess.Move.from_uci(str(previous_previous)))
                    move_chosen_previously1 = move_chosen_previously_previously + move_chosen_previously
                    probs = self.opening_book(canonicalBoard.board, book3[move_chosen_previously1])


                else:
                    copy_board = canonicalBoard.board.copy()
                    previous = copy_board.pop()
                    move_chosen_previously = copy_board.san(chess.Move.from_uci(str(previous)))
                    previous_previous = copy_board.pop()
                    move_chosen_previously_previously = copy_board.san(chess.Move.from_uci(str(previous_previous)))
                    previous_previous_previous = copy_board.pop()
                    move_chosen_previously_previously_previously = copy_board.san(
                        chess.Move.from_uci(str(previous_previous_previous)))
                    move_chosen_previously1 = move_chosen_previously_previously_previously + move_chosen_previously_previously + move_chosen_previously
                    probs = self.opening_book(canonicalBoard.board, book4[move_chosen_previously1])

            except Exception as x:
                print(x)
                for i in range(self.args.numMCTSSims):
                    dir_noise = (i == 0 and self.dirichlet_noise)
                    self.search(canonicalBoard, dirichlet_noise=dir_noise)

                    s = self.game.stringRepresentation(canonicalBoard)
                    counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

                if temp == 0:
                    bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
                    bestA = np.random.choice(bestAs)
                    probs = [0] * len(counts)
                    probs[bestA] = 1
                    return probs

                counts = [x ** (1. / temp) for x in counts]
                counts_sum = float(sum(counts))
                probs = [x / counts_sum for x in counts]



        else:

            for i in range(self.args.numMCTSSims):
                dir_noise = (i == 0 and self.dirichlet_noise)
                self.search(canonicalBoard, dirichlet_noise=dir_noise)

            s = self.game.stringRepresentation(canonicalBoard)
            counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

            if temp == 0:
                bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
                bestA = np.random.choice(bestAs)
                probs = [0] * len(counts)
                probs[bestA] = 1
                return probs

            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts))
            probs = [x / counts_sum for x in counts]

        return probs

    def search(self, canonicalBoard, dirichlet_noise=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if canonicalBoard.board.is_fifty_moves() or canonicalBoard.board.is_fivefold_repetition():
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
            if self.Es[s] != 0:
                # terminal node
                return -self.Es[s]

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node

            start = time.time()
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            end = time.time()
            self.time_nnet += (end - start)

            start = time.time()
            valids = self.game.getValidMoves(canonicalBoard, 1)

            end = time.time()
            self.time_vali += (end - start)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves

            start = time.time()
            if self.dirichlet_noise:
                self.applyDirNoise(s, valids)

            end = time.time()
            self.time_diri += (end - start)
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        start = time.time()
        if self.dirichlet_noise:
            self.applyDirNoise(s, valids)
            sum_Ps_s = np.sum(self.Ps[s])
            self.Ps[s] /= sum_Ps_s  # renormalize
        end = time.time()
        self.time_diri += (end - start)
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        start = time.time()
        # for a in range(self.game.getActionSize()):
        #   if valids[a]:
        for a in np.argwhere(valids):
            a = a[0]
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])

                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        end = time.time()
        self.time_pick += (end - start)

        a = best_act

        start = time.time()
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        end = time.time()
        self.time_next += (end - start)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

        return -v

    def applyDirNoise(self, s, valids):
        dir_values = np.random.dirichlet([self.args.dirichletAlpha] * np.count_nonzero(valids))
        dir_idx = 0
        s_policy = self.Ps[s]
        s_policy = np.argwhere(s_policy)  # optimization
        # for idx in range(len(self.Ps[s])):
        for idx in s_policy:
            idx = idx[0]
            if self.Ps[s][idx]:
                self.Ps[s][idx] = (0.75 * self.Ps[s][idx]) + (0.25 * dir_values[dir_idx])
                dir_idx += 1