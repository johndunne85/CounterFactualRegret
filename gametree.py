import numpy as np

ACTIONS = ['r', 'c', 'f']  # RAISE or CALL/CHECK or FOLD


class GameNode:
    def __init__(self, info_set):
        self.info_set = info_set
        self.strategy_sum, self.regret_sum = np.zeros((2, len(ACTIONS)))

    def update_strategy(self, opponent_probabilty=1):
        """same as the original regret matching algorithm except for haaving a weighing factor"""
        strategy = np.copy(self.regret_sum)
        strategy[strategy < 0] = 0  # reset negative regrets to zero

        summation = sum(strategy)
        if summation > 0:
            # normalise
            strategy /= summation
        else:
            # uniform distribution to reduce exploitability
            strategy = np.repeat(1 / len(ACTIONS), len(ACTIONS))

        self.strategy_sum += strategy * opponent_probabilty

        return strategy

    def avg_strategy(self):
        summation = sum(self.strategy_sum)
        avg_strategy = self.strategy_sum / summation if summation > 0 else np.repeat(1/len(ACTIONS), len(ACTIONS))
        return avg_strategy

    def __repr__(self):
        return '{i}: {a}'.format(i=self.info_set, a=self.avg_strategy())

class GameTree:
    '''Game tree where each node in the game tree performs regret matching'''
    def __init__(self):
        self.nodes = dict()

    def __getitem__(self, info_set):
        node = self.nodes.get(info_set)
        if node is None:
            node = GameNode(info_set)
            self.nodes[info_set] = node
        return node

    def __str__(self):
        return '\n'.join(map(str, self.nodes.values()))
