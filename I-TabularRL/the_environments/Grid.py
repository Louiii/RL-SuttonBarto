import numpy as np

class GridWorld():
    def __init__(self, w=5, h=5, actions=['up', 'right']):
        self.w=w
        self.h=h
        self.goal = (w-1, h-1)
        self.goals = [self.goal]
        self.actions=actions
        self.states=[(i, j) for i in range(w) for j in range(h)]
        cost = lambda i,j: (4+j-i)**2
        self.costs=np.array([[ cost(i,j) for j in range(h)] for i in range(w)])

    def step(self, state, action):
        """Return state and reward"""
        (i,j)=state
        if state==self.goal: return state, 0
        if (action=='right' and (i+1, j)==self.goal) or (action=='up' and (i, j+1)==self.goal):
            return self.goal, 100-self.costs[i,j]
        if action == 'right':
            if i==self.w-1:
                return state, -20-self.costs[i,j]
            return (i+1, j), -self.costs[i,j]
        elif action == 'up':
            if j==self.h-1:
                return state, -20-self.costs[i,j]
            return (i, j+1), -self.costs[i,j]

    def deterministicπEpisodeStillGoing(self, state, action):
        (i,j)=state
        if state==self.goal: return False
        if action=="right" and i==self.w-1: return False
        if action=="up"    and j==self.h-1: return False
        return True
