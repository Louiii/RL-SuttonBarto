class Maze:
    def __init__(self, blocking=False, shortcut=False):
        self.w = 9
        self.h = 6
        self.states = [(i,j) for i in range(self.w) for j in range(self.h)]
        self.actions = ["up", "down", "left", "right"]
        self.start = (0, 3)
        self.goals = [(8, 5)]
        self.obstacles = [(2,2),(2,3),(2,4), (5,1), (7,3),(7,4),(7,5)]
        self.time = 0
        self.blocking = blocking
        self.shortcut = shortcut
        if blocking: self.blockingMaze()
        if shortcut: self.shortcutMaze()

    def blockingMaze(self):
        self.start = (3,0)
        self.obstacles = [(i, 2) for i in range(8)]

    def shortcutMaze(self):
        self.start = (3,0)
        self.obstacles = [(i, 2) for i in range(1,9)]

    def changeEnvironment(self):
        if self.blocking: self.obstacles = [(i, 2) for i in range(1,9)]
        if self.shortcut: self.obstacles = [(i, 2) for i in range(1,8)]

    def step(self, state, action):
        self.time += 1
        if self.blocking and self.time == 1000: self.changeEnvironment()
        if self.shortcut and self.time == 3000: self.changeEnvironment()
        x, y = state
        if action == "up":
            y = min(y + 1, self.h - 1)
        elif action == "down":
            y = max(y - 1, 0)
        elif action == "left":
            x = max(x - 1, 0)
        elif action == "right":
            x = min(x + 1, self.w - 1)
        if (x, y) in self.obstacles:
            x, y = state
        if (x, y) in self.goals:
            reward = 1.0
        else:
            reward = 0.0
        return (x, y), reward
