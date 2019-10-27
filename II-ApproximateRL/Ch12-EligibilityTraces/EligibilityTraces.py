from modules import *

class ValueFn:
    def __init__(self, n_weights, α=0.05):
        self.θ = np.zeros(n_weights)
        self.n_weights = n_weights
        self.α = α

    def x(self, state):
        x_ = np.zeros(self.n_weights)
        if state not in [0, 20]:x_[state] = 1#for true online tdlambda
        # x_[state] = 1
        return x_

    def value(self, state):
        return self.θ[state]#np.dot(self.x(state), self.θ)

class SAR:
    def __init__(self, S=None, A=None, R=None):
        self.S = S
        self.A = A
        self.R = R

class predAgent:
    def __init__(self, env, n_weights, γ=1, λ=0.9, α=0.05):
        self.env = env
        self.V = ValueFn(n_weights, α)
        self.γ = γ# discount factor
        self.λ = λ# trace decay rate

    def G(self, rewards, s_tn, t, n):
        G = 0
        λi = 1
        for i in range(n):#(1, n+1):
            G += rewards[t+i]*λi # G += rewards[t+i]*self.λ**(i-1)
            λi *= self.λ
            if λi < 1e-3: break
        return G + self.V.value(s_tn)*self.λ**n

    def Gλ(self, trajectory, t, T):
        G1 = 0
        λi = 1
        rewards = [x.R for x in trajectory]
        for n in range(1, T-t):
            G1 += self.G(rewards, trajectory[t+n].S, t, n)*λi
            λi *= self.λ
            if λi < 1e-3: break
        return (1 - self.λ)*G1 if λi < 1e-3 else sum(rewards)*λi + (1 - self.λ)*G1#self.λ**(T-t-1) + (1 - self.λ)*G1

    def OfflineSemiGradientλReturn_episode(self):# needs fixing!!
        state = self.env.start
        trajectory = []
        T = 0
        while state not in self.env.goals:
            T += 1
            action = np.random.choice(self.env.actions)#self.Π(state)
            next_state, reward = self.env.step(state, action)
            trajectory.append(SAR(state, action, reward))#state)
            state = next_state

        for t in range(T):
            δ = self.Gλ(trajectory, t, T) - self.V.value(trajectory[t].S)#trajectory[t])self.Gλ(rewards, trajectory, t, T)
            self.V.θ += self.V.α * δ * self.V.x(trajectory[t].S)


    def semiGradientTDλ_episode(self, trace="accumulating traces"):
        state = self.env.start
        z = np.zeros(self.V.n_weights)
        while state not in self.env.goals:
            action = np.random.choice(self.env.actions)#self.Π(state)
            next_state, reward = self.env.step(state, action)
            if trace=="accumulating traces":
                z = self.γ * self.λ * z + self.V.x(state)# x(s) is the grad of lin apx
            else:
                z *= self.γ * self.λ
                if state not in self.env.goals: z[state] = 1

            δ = reward + self.γ * self.V.value(next_state) - self.V.value(state)
            self.V.θ += self.V.α * δ * z
            state = next_state

    def trueOnlineTD(self):
        """ Online n-step truncated Semi-Gradient TD λ-return. """
        state = self.env.start
        x = self.V.x(state)
        z = np.zeros(self.V.n_weights)
        V_old = 0
        while state not in self.env.goals:
            action = np.random.choice(self.env.actions)#self.Π(state)
            next_state, reward = self.env.step(state, action)
            next_x = self.V.x(next_state)
            state_V = self.V.value(state)
            next_state_V = self.V.value(next_state)

            δ = reward + self.γ * next_state_V - state_V
            z *= self.γ * self.λ
            z += (1 - self.V.α * self.γ * self.λ * np.dot(z, x)) * x# x(s) is the grad of lin apx

            self.V.θ += self.V.α * (δ + state_V - V_old) * z - self.V.α * (state_V - V_old) * x

            state = next_state
            x = next_x
            V_old = next_state_V

    def monteCarloDutchTraces(self):
        state = self.env.start
        trajectory = [state]
        G = 0
        while state not in self.env.goals:
            action = np.random.choice(self.env.actions)#self.Π(state)
            next_state, reward = self.env.step(state, action)
            state = next_state
            trajectory.append(state)

        a = self.V.θ
        z = self.V.x(trajectory[0])
        for t in range(len(trajectory)):
            #LINEAR MC self.V.θ += self.V.α * ( G - self.V.value(states[t]) ) * self.V.x(states[t])
            x = self.V.x(trajectory[t])
            self.V.θ = a + self.V.α * G * z
            z += (1 - self.V.α * np.dot( z, x )) * x
            a -= self.V.α * np.dot( a, x ) * x


