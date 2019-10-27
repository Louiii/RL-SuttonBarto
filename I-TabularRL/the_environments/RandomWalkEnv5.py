import numpy as np

class Walk():
    def __init__(self):
        self.N_STATES = 5
        self.actions = ['right', 'left']

        self.states = np.arange(1, self.N_STATES + 1) # all states but terminal states
        # self.states = np.arange(0, self.N_STATES + 2)

        self.start = 3 # start from the middle state
        self.goals = [0, self.N_STATES + 1]
        self.goal = self.N_STATES + 1

        # true state value from bellman equation
        self.true_V = np.array([i/6 if i!=6 else 0 for i in range(7)])

    def step(self, state):
        next_state = state + 1 if np.random.random()<0.5 else state - 1
        # if next_state == 0: return next_state, 0
        if next_state == self.goal: return next_state, 1
        return next_state, 0
