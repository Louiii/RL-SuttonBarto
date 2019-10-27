from modules import *

class Π:
    def __init__(self, αθ, λθ, n, actions):
        self.α = αθ
        self.λ = λθ
        self.θ = np.zeros(n)#np.random.randn(n)*0.05
        self.n = n
        self.actions = actions

class V:
    def __init__(self, αw, λw, n):
        self.α = αw
        self.λ = λw
        self.w = np.zeros(n)
        self.n = n

class EpisodicAgent:
    def __init__(self, env, γ, π_args, v_args):
        """ args:
        π_args = [αθ, λθ, n]       - where n is the number of states action pairs
        v_args = [αw, λw, n]       - where n is the number of states
                                   - λs are None for first fn
        """
        self.env = env
        self.γ = γ
        self.π = Π(*(π_args+[env.actions]))
        self.v = V(*v_args)

    def onestep(self):
        """
        The natural state-value-function learning method to pair with this
        would be semi-gradient TD(0).
        """
        state = self.env.start
        I = 1
        while state not in self.env.goals:
            action = self.π.sample(state)
            next_state, reward = self.env.step(state, action)
            δ = reward + self.γ*self.v.value(next_state) - self.v.value(state)
            self.v.w += self.v.α*δ*self.v.grad(state)
            self.π.θ += self.π.α*I*δ*self.π.gradLn(state, action)
            I *= self.γ
            state = next_state
        # return

    def eligibilityTraces(self):
        state = self.env.start
        I = 1
        zθ, zw = np.zeros(self.π.n), np.zeros(self.v.n)
        while state not in self.env.goals:
            action = self.π.sample(state)
            next_state, reward = self.env.step(state, action)

            δ = reward + self.γ*self.v.value(next_state) - self.v.value(state)

            zw += self.γ*self.v.λ*zw + self.v.grad(state)
            zθ += self.γ*self.π.λ*zθ + I*self.π.gradLn(state, action)

            self.v.w += self.v.α*δ*zw
            self.π.θ += self.π.α*δ*zθ

            I *= self.γ
            state = next_state

class ContinuingAgent:
    def __init__(self, env, α_R, π_args, v_args):
        self.env = env
        self.α_R = α_R
        self.π = Π(*(π_args+[env.actions]))
        self.v = V(*v_args)

        self.R = 0 # average reward
        self.zθ, self.zw = np.zeros(self.π.n), np.zeros(self.v.n)
        # self.state = env.start


    def eligibilityTraces_timestep(self, state):
        """ backward view """
        action = self.π.sample(state)#self.state)
        next_state, reward = self.env.step(state, action)#self.state, action)

        δ = reward - self.R + self.v.value(next_state) - self.v.value(state)#self.state)
        self.R += self.α_R*δ

        self.zw += self.v.λ*self.zw + self.v.grad(state)#self.state)
        self.zθ += self.π.λ*self.zθ + self.π.gradLn(state, action)#self.state, action)

        self.v.w += self.v.α*δ*zw
        self.π.θ += self.π.α*δ*zθ

        state = next_state#self.state = next_state
