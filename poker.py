import itertools
from collections import Counter

# example string representation of card '8 Diamond': '8D'
RANKS = [str(i) for i in range(2, 11)] + ['J', 'Q', 'K', 'A']
SUITS = ['C', 'D', 'H', 'S']
POKER_BOX = [r+s for r,s in itertools.product(RANKS, SUITS)]

class Card:
    def __init__(self, card):
        self.suit = card[-1]
        self.value = self.determine_value(card[:-1])

    def determine_value(self, x):
        try:
            return int(x)
        except ValueError:
            return {
                'J': 11, 'Q': 12, 'K': 13, 'A': 14
            }.get(x)

    def __repr__(self):
        return '{}_{}'.format(self.value, self.suit)


class Hand:
    def __init__(self, cards):
        self.cards = [Card(c) for c in cards]
        self.values = [card.value for card in self.cards]
        self.values.sort()
        self.counts = Counter(self.values)

    def is_flush(self):
        currentSuit = self.cards[0].suit
        for card in self.cards[1:]:
            if card.suit != currentSuit:
                return False
        return True#, self.values

    def is_royal_flush(self):
        if self.values == range(10,15) and self.is_flush():
            return True
        else:
            return False

    def is_four_of_a_kind(self):
        if 4 in self.counts.values():
            return True
        return False

    def is_three_of_a_kind(self):
        if 3 in self.counts.values():
            return True
        return False

    def is_straight(self):
        low = min(self.values)
        if self.values == range(low,low+5) or self.values == [2,3,4,5,14]:
            return True
        else:
            return False

    def is_full_house(self):
        if 2 in self.counts.values() and 3 in self.counts.values():
            return True
        return False

    def is_two_pair(self):
        if all(x[1] == 2 for x in self.counts.most_common()[:2]):
            return True
        return False

    def is_pair(self):
        if 2 in self.counts.values() and len(self.counts) == 4:
            return True
        return False

    def get_score(self):
        if self.is_royal_flush():                      return (9, 'royal_flush')
        elif self.is_flush() and self.is_straight(): return (8, 'straight_flush')
        elif self.is_four_of_a_kind():                  return (7, 'four_of_a_kind')
        elif self.is_full_house():                      return (6, 'full_house')
        elif self.is_flush():                          return (5, 'flush')
        elif self.is_straight():                      return (4, 'straight')
        elif self.is_three_of_a_kind():              return (3, 'three_of_a_kind')
        elif self.is_two_pair():                      return (2, 'two_pair')
        elif self.is_pair():                          return (1, 'pair')
        else:                                          return (0, 'high_card')
