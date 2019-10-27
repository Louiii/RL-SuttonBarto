import numpy as np

class ServersEnvironment:
    def __init__(self):
        self.priorities = np.arange(0, 4)# possible priorities
        self.rewards = np.power(2, np.arange(0, 4))# reward for each priority
        self.actions = ['reject', 'accept']# reject, accept
        self.n_servers = 10
        self.p_free = 0.06# at each time step prob of becoming free
        self.available_servers = 10
        self.start = (np.random.choice(self.priorities), self.n_servers)

    def step(self, state, action):
        """ state = (priority of the current customer, number of free servers)"""
        customer, self.available_servers = state

        if action=='accept' and self.available_servers > 0:
            reward = self.rewards[customer]
            self.available_servers -= 1
        else:
            reward = 0

        # some servers become available:
        self.available_servers += np.random.binomial(self.n_servers-self.available_servers, self.p_free)

        # arrivals:
        next_customer = np.random.choice(self.priorities)

        next_state = (next_customer, self.available_servers)
        return next_state, reward
