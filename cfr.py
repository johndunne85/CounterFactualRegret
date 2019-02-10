import random
from poker import POKER_BOX, Hand, Card
from gametree import GameTree, ACTIONS
import numpy as np
import pandas as pd
import itertools
import copy
from collections import Counter
'''
	Use conterfactual regret minimisation algorithm to play Five card stud Poker
	Counterfactual regret minimization uses the regret-matching algorithm. In addition,
	(1)one must additionally factor in the probabilities of reaching each information set given the players'
	strategies, and (2) given that the game is treated sequentially through a sequence of information sets,
	there is a passing forward of game state information and probabilities of player action sequences, and
	a passing backward of utility information through these information sets
'''

class PokerGame:
    def __init__(self):
        random.seed()
        self.game_tree = GameTree()

    def __str__(self):
        return 'history={};ante={};pot={};shared_cards={}'.format(self.history, self.ante, self.pot, self.shared_cards)

    def reset(self, *players):
        random.shuffle(POKER_BOX)
        self.poker_box = copy.copy(POKER_BOX)

    def deal(self, n=1):
        return [self.poker_box.pop() for i in range(n)]

    def new_game(self, n_players=2):
        self.reset()
        cards = {i:[] for i in range(n_players)}
        for i in range(5*n_players):
            cards[i%n_players] += self.deal()
        return list(cards.values())

    def calculate_pot(self, history, pot=50, bet=20):
        """calculate the pot for a given history of a two player game
        """
        call_needed = False
        for char in history:
            if char == 'c' and call_needed:
                    pot += bet
            if char == 'r':
                if call_needed:
                    pot += bet
                call_needed = True
                pot += bet
            if char == 'f':
                break
        return pot

    def end(self, history, cards, verbose=False):
        """calculate the reward and pot of a two player game
        """
        def best_hand(cards):
            # not efficient but for the sake of readability...
            all_hands = list(itertools.permutations(cards , 5))
            # all_hands = itertools.permutations(cards + self.shared_cards, 5)
            scores = [Hand(hand).get_score() for hand in all_hands]
            return max(scores),Hand(all_hands[scores.index(max(scores))])

        p1_cards, p2_cards = cards
        pot = self.calculate_pot(history)
        # case 1: someone has folded
        if 'f' in history:
            folding_player = history.index('f')
            winning_player = (folding_player + 1)%2
            return (winning_player, pot)

        # case 2: entered show-down stage, needs to evaluate hands
        p1_best, p1_hand = best_hand(p1_cards)
        p2_best, p2_hand = best_hand(p2_cards)
        if verbose:
            print('P1:{}, P2:{}'.format(p1_best, p2_best))

        # winner gets all
        if p1_best > p2_best:
            return 0, pot
        elif p1_best < p2_best:
            return 1, pot
        else:
            #when both hands are same, consider values of the cards
            p1vals = list(pd.Series(Counter(p1_hand.values)).sort_values().index)
            p2vals = list(pd.Series(Counter(p2_hand.values)).sort_values().index)

            for i in range(len(p1vals)):
                if p1vals[-i] > p2vals[-i]:
                    return 0, pot
                if p2vals[-i] > p1vals[-i]:
                    return 1, pot
            return 0.5, pot/2  # players share the pots

    def cfr(self, cards, history, p1_prob, p2_prob, max_actions=4):
        '''
        returns the utility of the current node, which is determined by the history and the cards.
        :param cards: p1 gets cards[0] and p2 gets cards[1]
        :param history: sequence of players' actions
        :param p1_prob: probability that player1's strategy can reach this history
        :param p2_prob: probability that player2's strategy can reach this history
        :param max_actions: integer max number of actions of history
        :return:
        '''
        # decide who is playing
        plays = len(history)
        player = plays % 2
        opp = 1 - player

        if plays > 1:
            # check whether game has terminated.
            if len(history) >= max_actions or 'f' in history or (len(history) > 1 and history[-1]=='c'):
                winner, utility = self.end(history, cards)
                return utility if not winner or winner is player else -utility

        # get/create the node corresponding to the info set
        info_set = ''.join(sorted(cards[player], reverse=True)) + history
        node = self.game_tree[info_set]

        # use regret matching to update strategy
        strategy = node.update_strategy(p2_prob if player == 0 else p1_prob)

        # for each action, recursively call cfr with additional history and corresponding player probabilities
        action_utils = np.zeros(len(ACTIONS))
        for i, action in enumerate(ACTIONS):
            next_history = history + action
            res = self.cfr(cards, next_history, p1_prob * strategy[i], p2_prob) if player == 0 else \
                  self.cfr(cards, next_history, p1_prob, p2_prob * strategy[i])

            action_utils[i] = -res

        # for each action, compute and accumulate counterfactual regret
        # weighted by the probability that the opponent plays to the current information set,
        node_util = sum(action_utils * strategy)
        regret = action_utils - node_util
        node.regret_sum += regret * (p2_prob if player == 0 else p1_prob)

        return node_util

    def start(self, games):
        """run cfr for a set number of games and print the outcome of the game tree afterwards"""
        for g in range(games):
            cards = self.new_game()
            self.cfr(cards, '', 1, 1)
        print(self.game_tree)

    def same_game(self, games):
        """run CFR over many games with the same deal of cards to see how the nash equilibrium converges for that set of hands"""
        cards = self.new_game()
        for g in range(games):
            self.cfr(cards, '', 1, 1)
        print(self.game_tree)

if __name__ == '__main__':
    PokerGame().start(1000)
