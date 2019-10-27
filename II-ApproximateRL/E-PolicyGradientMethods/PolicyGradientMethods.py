from modules import *

class Π:
    def __init__(self, n, actions):
        self.θ = np.array([-1.47, 1.47])#np.zeros(n)#np.random.randn(n)*0.05
        self.n = n
        self.actions = actions

    def h(self, state, action):
        """ could be an ANN """
        return np.dot(self.θ, self.x(state, action))

    def sample(self, state):
        sm = sum([np.exp(self.h(state, b)) for b in self.actions])
        actions = {a:np.exp(self.h(state, a))/sm for a in self.actions}
        # actions = {a:self.π(state, a) for a in self.action}
        return np.random.choice(list(actions.keys()), p=list(actions.values()))

    def π(self, state, action):
        return np.exp(self.h(state, action)) / sum([np.exp(self.h(state, b)) for b in self.actions])

    def gradLn(self, state, action):
        sm = sum([self.π(state, b)*self.x(state, b) for b in self.actions])
        return self.x(state, action) - sm

    def x(self, state, action):
        x_ = np.zeros(self.n)
        # if state==3:
        #     x_[2] = 1# goal state
        # el
        if action=="left":
            x_[0] = 1
        else:
            x_[1] = 1
        return x_

class V:
    def __init__(self, αw, n):
        self.α = αw
        self.w = np.zeros(n)
        self.n = n

    def value(self, state):
        return np.dot(self.x(state), self.w)

    def grad(self, state):
        return self.x(state)

    def x(self, state):
        x_ = np.zeros(self.n)
        x_[0] = 1
        return x_

class Agent:
    def __init__(self, env, α, γ, n, nαw=None):
        self.env = env
        self.α = α
        self.γ = γ
        self.π = Π(n, env.actions)
        if nαw is not None: self.v = V(*nαw)

    def REINFORCE(self):
        """ Monte-Carlo Policy-Gradient Control (episodic) for π*

        I index my traj as SAR_0, SAR_1,...
        not like in the book (page 328): S_0 A_0 R_1, S_1 A_1 R_2, ..."""

        # Episode:
        state = self.env.start
        trajectory = []
        T = 0
        G0 = 0
        while state not in self.env.goals:
            T += 1
            action = self.π.sample(state)
            next_state, reward = self.env.step(state, action)
            trajectory.append( (state, action, reward) )
            state = next_state
            G0 += reward

        # Learn:
        for t in range(T):
            G = sum([trajectory[k][2]*self.γ**(k-t) for k in range(t, T)])
            self.π.θ += self.α*(self.γ**t)*G*self.π.gradLn(trajectory[t][0], trajectory[t][1])

        return G0

    def baselineREINFORCE(self):
        """ Monte-Carlo Policy-Gradient Control (episodic) for π*

        I index my traj as SAR_0, SAR_1,...
        not like in the book (page 328): S_0 A_0 R_1, S_1 A_1 R_2, ..."""

        # Episode:
        state = self.env.start
        trajectory = []
        T = 0
        G0 = 0
        while state not in self.env.goals:
            T += 1
            action = self.π.sample(state)
            next_state, reward = self.env.step(state, action)
            trajectory.append( (state, action, reward) )
            state = next_state
            G0 += reward

        # Learn:
        for t in range(T):
            G = sum([trajectory[k][2]*self.γ**(k-t) for k in range(t, T)])
            δ = G - self.v.value(trajectory[t][0])
            self.v.w += self.v.α * δ * self.v.grad(trajectory[t][0])
            self.π.θ += self.α*(self.γ**t)*δ*self.π.gradLn(trajectory[t][0], trajectory[t][1])

        return G0
