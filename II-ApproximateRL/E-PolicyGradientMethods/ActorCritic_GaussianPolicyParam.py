from modules import *

class Π:
    """ Gaussian Policy Parameterisation
    the feature encoding functions: xμ and xσ can be changed.
    """
    def __init__(self, αθ, λθ, n, actions):# n must be even
        self.α = αθ
        self.λ = λθ
        self.θ = np.zeros(n)# θ has first m elems for μ, second m for σ, it doesn't have to be like this!
        self.n = n
        self.m = n/2
        self.actions = actions

    def θμ(self):
        return self.θ[:self.m]

    def θσ(self):
        return self.θ[self.m:]

    def xμ(self, state):
        x = np.zeros(self.m)

        return

    def xσ(self, state):
        x = np.zeros(self.m)

        return

    def μ(self, state):
        return np.dot(self.θμ(), self.xμ(state))

    def σ(self, state):
        return np.exp( np.dot(self.θσ(), self.xσ(state)) )

    def π(self, state, action):
        """ single continuous action """
        c = 1/(self.μ(state)*np.sqrt(2*np.pi))
        expt = -( (action - self.μ(state))**2 )/( 2*self.σ(state)**2 )
        return c * np.exp( expt )

    def gradLn_wrt_μ(self, state, action):
        return self.xμ(state)*(action - self.μ(state))/self.σ(state)**2

    def gradLn_wrt_σ(self, state, action):
        return self.xσ(state)*(((action - self.μ(state))**2)/self.σ(state)**2 - 1)

    def gradLn(self, state, action):
        """ I think?! """
        return np.concatenate( (self.gradLn_wrt_μ(state, action), self.gradLn_wrt_σ(state, action)) )

class V:
    def __init__(self, αw, λw, n):
        self.α = αw
        self.λ = λw
        self.w = np.zeros(n)
        self.n = n

class ContinuingAgent:
    def __init__(self, env, α_R, π_args, v_args):
        self.env = env
        self.α_R = α_R
        self.π = Π(*(π_args+[env.actions]))
        self.v = V(*v_args)

        self.R = 0 # average reward
        self.zθ, self.zw = np.zeros(self.π.n), np.zeros(self.v.n)

    def eligibilityTraces_timestep(self, state):
        """ backward view """
        action = self.π.sample(state)
        next_state, reward = self.env.step(state, action)

        δ = reward - self.R + self.v.value(next_state) - self.v.value(state)
        self.R += self.α_R*δ

        self.zw += self.v.λ*self.zw + self.v.grad(state)
        self.zθ += self.π.λ*self.zθ + self.π.gradLn(state, action)

        self.v.w += self.v.α*δ*zw
        self.π.θ += self.π.α*δ*zθ

        state = next_state
        return state

env = SomeEnv()
state = env.start

agent = ContinuingAgent(..)
def episode():
    while True:
        state = agent.eligibilityTraces_timestep(state)
        if state ...: break
while True:
    episode()
    if ... : break
