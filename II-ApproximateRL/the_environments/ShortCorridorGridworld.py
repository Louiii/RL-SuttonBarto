

class SCGridWorld:
    def __init__(self):
        self.actions = ["left", "right"]
        self.states = [0,1,2,3]
        self.goals = [3]
        self.start = 0

    def step(self, state, action):
        if state==0 and action=="left": return 0, -1
        if state==0 and action=="right": return 1, -1
        if state==1 and action=="left": return 2, -1
        if state==1 and action=="right": return 0, -1
        if state==2 and action=="left": return 1, -1
        if state==2 and action=="right": return 3, 0
        return 3, 0
