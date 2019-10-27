import numpy as np

class Walk():
    def __init__(self):
        self.N = 1000
        self.actions = ['right', 'left']

        self.states = np.arange(1, self.N + 1) # all states but terminal states
        # self.states = np.arange(0, self.N + 2)

        self.start = 500 # start from the middle state
        self.goals = [0, self.N + 1]

        self.STEP_RANGE = 100
        # # true state value from bellman equation
        self.true_V = None#self.compute_true_value()
        # self.true_V[0] = self.true_V[-1] = 0

    def compute_true_value(self):
        # true state value, just a promising guess
        true_value = np.arange(-1001, 1003, 2) / 1001.0

        # Dynamic programming to find the true state values, based on the promising guess above
        # Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states
        while True:
            old_value = np.copy(true_value)
            for state in self.states:
                true_value[state] = 0
                for action in self.actions:
                    for step in range(1, self.STEP_RANGE + 1):
                        a = 1
                        if action=='left': a = -1
                        next_state = state + step * a
                        next_state = max(min(next_state, self.N + 1), 0)
                        # asynchronous update for faster convergence
                        true_value[state] += 1.0 / (2 * self.STEP_RANGE) * true_value[next_state]
            error = np.sum(np.abs(old_value - true_value))
            if error < 1e-2: break
        true_value[0] = true_value[-1] = 0

        return true_value

    def step(self, state, action):
        step = np.random.randint(1, self.STEP_RANGE + 1)
        a = 1
        if action=='left':
            a = -1
        state += step * a
        state = max(min(state, self.N + 1), 0)
        if state == 0:
            reward = -1
        elif state == self.N + 1:
            reward = 1
        else:
            reward = 0
        return state, reward
