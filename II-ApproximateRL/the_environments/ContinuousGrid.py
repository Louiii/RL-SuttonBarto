import numpy as np

class GridWorldCts():
    def __init__(self, w=6, h=5, actions=['up', 'right'], start=(0,0),
                means=[[1, 1],[2, 2],[3, 3],[4, 3],[0, 4],[1, 4],[4, 0],[4, 1],[5, 0],[5, 1],[2, 3]],

                    ):
        self.w=w
        self.h=h
        self.start=start
        self.goal = (w-1, h-1)
        self.goals = [self.goal]

        self.stepSize = 0.45
        self.actions=actions
        self.states=[(i, j) for i in range(w) for j in range(h)]
        self.means = means

        # self.mask = self.newMask()
        # self.costs_average=np.zeros(shape=(w,h))

    def goalState(self, state):
        (x, y), (gx, gy) = state, self.goal
        if x > gx and y > gy: return True
        return False

    def performAction(self, state, action):
        (x, y) = state
        if action=='right':
            return ( min(x+self.stepSize, self.w), y )
        if action=='up':
            return ( x, min(y+self.stepSize, self.h) )

    def hitWall(self, state, action):
        (x, y) = state
        if action=='right' and x+self.stepSize > self.w:
            return True
        if action=='up' and y+self.stepSize > self.h:
            return True
        return False

    def step(self, state, action):
        """Return state and reward"""
        (x,y)=state
        if self.goalState(state):
            return state, 0
        next_state = self.performAction(state, action)
        if self.goalState(next_state):
            return next_state, 100-self.costAll(next_state)
        if self.hitWall(state, action):
            return next_state, -20-self.costAll(next_state)
        return next_state, -self.costAll(next_state)

    # def newMask(self):
    #     self.mask = [k for k in range(len(means)-1) if random.random() < 0.5]
    #     if random.random() < 0.05:
    #         self.mask.append( len(self.means) )
    #     # return self.mask
    #
    # def cost(self, state, nMask=False):
    #     if nMask:
    #         self.newMask()
    #     (i, j) = state
    #     cov = np.matrix([[0.3, 0], [0, 0.3]])
    #     c=sum([norm_pdf_multivariate(np.array([i, j]), np.array(self.means[k]), cov) for k in range(len(self.means)) if k in self.mask])
    #     return 100*c

    def costAll(self, x, y):
        c = 0
        for mu in self.means:
            c += np.exp(-0.5 * (10/3 * ((x-mu[0])**2 + (y-mu[1])**2))) / (np.pi * 0.6 )
        return 100*c

    def deterministicπEpisodeStillGoing(self, state, action):
        (i,j)=state
        if state==self.goal: return False
        if action=="right" and i==self.w-1: return False
        if action=="up"    and j==self.h-1: return False
        return True
