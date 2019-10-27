import numpy as np

class PW:
    def __init__(self):
        self.wind = 0.5
        self.map=[  [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 2, 2, 2, 2, 1, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 2, 3, 2, 2, 1, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 2, 3, 2, 1, 1, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  ]

        self.states = list(range(len(self.map)*len(self.map[0])))
        self.actions = list(range(4))
        start_states = [ (6, 1), (7, 1), (11, 1), (12, 1) ]
        self.start = [ (6, 1), (7, 1), (11, 1), (12, 1) ][ np.random.randint( 0, 3 ) ]
        # self.done = False


    def checkValid(self, row, col):
        valid=False
        numRows=len(self.map)
        numCols=len(self.map[0])

        if(row < numRows and row >= 0 and col < numCols and col >= 0):
            if self.map[row][col] != 5:#obstacle
                valid=True
        return valid

    def step(self, state, action):
        r, c = state
        # Stochastic actions
        other_actions = set(range(4))
        other_actions.remove(action)
        other_actions = list(other_actions)
        if np.random.random() < 0.1:
            action = other_actions[ np.random.randint(0, 2) ]

        newr, newc = r, c
        if action == 0: newr += 1
        if action == 1: newr -= 1
        if action == 2: newc -= 1
        if action == 3: newc += 1

        #Check if new position is out of bounds or inside an obstacle
        if self.checkValid(newr, newc): r, c = newr, newc

        if np.random.random() < self.wind:
            newc = c + 1
            if self.checkValid(newr, newc): r, c = newr, newc

        # REWARDS:
        reward = 0
        if self.map[r][c] == 4:# Goal
            reward = 10
            # self.done = True
        if self.map[r][c] in [1,2,3]: reward = -self.map[r][c]
        return (r, c), reward

    def goal(self, state):
        r, c = state
        if self.map[r][c] == 4:# Goal
            return True
        return False
