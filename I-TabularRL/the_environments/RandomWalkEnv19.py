import numpy as np

class Walk():
    def __init__(self):
        self.N_STATES = 19
        self.actions = ['right', 'left']

        self.states = np.arange(1, self.N_STATES + 1) # all states but terminal states
        # self.states = np.arange(0, self.N_STATES + 2)

        self.start = 10 # start from the middle state
        self.goals = [0, self.N_STATES + 1]

        # true state value from bellman equation
        self.true_V = np.arange(-20, 22, 2) / 20.0
        self.true_V[0] = self.true_V[-1] = 0

    def step(self, state, action):
        if action=='right':
            next_state = state + 1
        else:
            next_state = state - 1

        if next_state == 0:
            return next_state, -1
        elif next_state == 20:
            return next_state, 1
        return next_state, 0
