import numpy as np
from scipy.stats import poisson

class CarRental:
    def __init__(self):
        self.w, self.h = 21, 21
        self.location1 = 0
        self.location2 = 0
        # state is (num cars 1st loc, num cars 2nd loc)
        self.states  = [(i, j) for i in range(self.w) for j in range(self.h)]
        self.actions = list(range(-5, 6, 1))

    def returns_and_rentings(self, state):
        # renting first.., then returns:
        loc1, loc2 = state
        to_rent_from_1, to_rent_from_2 = np.random.poisson(3), np.random.poisson(4)
        reward1 = to_rent_from_1 if to_rent_from_1 < loc1 else loc1
        reward2 = to_rent_from_2 if to_rent_from_2 < loc2 else loc2
        loc1, loc2 = loc1 - reward1, loc2 - reward2
        # returns:
        returns_to_1, returns_to_2 = np.random.poisson(3), np.random.poisson(2)
        loc1 = min(20, loc1+returns_to_1)
        loc2 = min(20, loc2+returns_to_2)
        return (loc1, loc2), (reward1 + reward2)*10


    def expected_reward(state, action):

    def p_next_state(self, state):
        ''' return 2 dictionarys of probabilities, rewards of all possible next states and their probs. '''
        self.location1, self.location2 = state

        p = {}
        for a in self.actions:
            prob = poisson.pmf(num, mean)

            p[next_state] = None
        return p

    def step(self, state, action):
        ''' how many car are to be moved overnight. '''
        # update state: 
        # action = num of cars to be moved from loc1 to loc2
        self.location1, self.location2 = state

        if action > self.location1: 
            self.location2 = min(20, self.location2 + self.location1)
            self.location1 = 0
        elif action < -self.location2:
            # more cars than loc 2 has to offer
            self.location1 = min(20, self.location2 + self.location1)
            self.location2 = 0
        else:
            # loc 1 or loc 2 have enough cars to offer
            # but they may still offer too much!
            if action > 0:# goes from loc 1 to loc 2
                self.location1 -= action
                self.location2 = min(20, self.location2+action)
            else:# goes from loc 2 to loc 1
                self.location2 += action# action is negative
                self.location1 = min(20, self.location2-action)

        next_state, reward = self.returns_and_rentings()
        return next_state, reward - np.abs(action)*2