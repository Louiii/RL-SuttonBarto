from modules import *

class Node():
    def __init__(self):
        self.visits=0
        self.q=0

class SAR():
    def __init__(self):
        self.S = None
        self.A = ""
        self.R = 0.0
    def SA(self):
        return (self.S, self.A)

class Agent():
    def __init__(self, env, γ=0.9, α=0.5, ε=0.1, n=3, V=None, iterations_to_record=[]):#, sm="εGreedy"):
        self.env = env
        self.γ = γ
        self.α = α
        self.ε = ε
        self.n = n
        self.iterations_to_record = iterations_to_record
        # self.selectionMethod = sm

        self.π = {s: np.random.choice(env.actions) for s in env.states}
        self.Π = {(s,a): 1/len(env.actions) for s in env.states for a in env.actions}
        self.V = {s: np.random.randn() for s in env.states} if V is None else V
        # for goal in env.goals: self.V[goal] = 0
        # self.V[env.goal] = 0

        self.S = {i: None for i in range(n+1)}
        self.R = {i: 0 for i in range(n+1)}

        self.SARs = {i: SAR() for i in range(n+1)}
        self.Q = {(s,a): np.random.randn() for s in env.states for a in env.actions}
        # for a in env.actions: self.Q[(env.goal,a)] = 0
        # self.Q2 = {(s,a): np.random.randn() for s in env.states for a in env.actions}
        # for a in env.actions: self.Q2[(env.goal,a)] = 0

    def sampleΠ(self, state):# choose an action over prob dist Π
        pmf = {a: self.Π[(state,a)] for a in self.env.actions}
        return np.random.choice(list(pmf.keys()), p=list(pmf.values()))

    def maxAction(self, s):# returns the action with the highest Q-value
        Qs=[(action, self.Q[(s,action)]) for action in self.actions]
        mx = max(Qs, key=lambda x:x[1])[1]
        mxlt = [a for (a,q) in Qs if q==mx]
        return mxlt[np.random.randint(len(mxlt))]

    def updateπ(self):
        for s in self.env.states: self.π[s] = self.maxAction(s)

    def generateSoftPolicy(self, ε=None):
        if ε==None: ε=self.ε
        self.updateπ()
        for s in self.env.states:
            for a in self.env.actions:
                if a == self.π[s]:
                    self.Π[(s,a)] = 1 - ε + ε/len(self.env.actions)
                else:
                    self.Π[(s,a)] = ε/len(self.env.actions)
    """ Algo 1: """
    def nStepTD_VPredictionEpisode(self, start=None, rnd_policy=False):
        self.S[0] = start if start is not None else self.env.states[np.random.randint(len(self.env.states)-1)]
        T = np.inf
        t, τ = 0, 0
        while τ != T-1:
            if t < T:
                state = self.S[t%(self.n+1)]
                action = self.sampleΠ(state) if rnd_policy==False else np.random.choice(self.env.actions)

                next_state, reward = self.env.step(state, action)

                # Store in buffer...
                self.S[(t+1)%(self.n+1)] = next_state
                self.R[(t+1)%(self.n+1)] = reward

                if next_state in self.env.goals:#self.env.deterministicπEpisodeStillGoing(self.S[t%(self.n+1)], action)==False:
                    T = t+1 #state!=self.env.goal
            τ = t - self.n + 1
            if τ >= 0:
                G = sum([self.R[i%(self.n+1)] * self.γ**(i-τ-1) for i in range(τ+1, min(τ+self.n, T)+1)])
                if τ + self.n < T:
                    G += self.V[self.S[(τ+self.n)%(self.n+1)]] * self.γ**self.n
                self.V[self.S[τ%(self.n+1)]] += self.α * (G - self.V[self.S[τ%(self.n+1)]])
            t+=1

    """ Algo 2: """
    def nStepOnPolicySARSA_Control_episode(self):
        # Randomly select starting state:
        self.SARs[0].S = self.env.states[np.random.randint(len(self.env.states)-1)]
        self.SARs[0].A = self.sampleΠ(self.SARs[0].S)
        T = np.inf
        t, τ = 0, 0
        while τ != T-1:
            if t < T:
                state = self.SARs[t%(self.n+1)].S
                action = self.SARs[t%(self.n+1)].A
                # take action, store resultant state and reward at next timestep.
                next_state, reward = self.env.step(state, action)

                self.SARs[(t+1)%(self.n+1)].S = next_state
                self.SARs[(t+1)%(self.n+1)].R = reward
                # if this next state is terminal:T = t+1 else: choose next action from policy
                if next_state in self.env.goals:#self.env.deterministicπEpisodeStillGoing(self.SARs[t%(self.n+1)].S, self.SARs[t%(self.n+1)].A)==False: #state!=self.env.goal
                    T = t+1
                else:
                    self.SARs[(t+1)%(self.n+1)].A = self.sampleΠ(self.SARs[(t+1)%(self.n+1)].S)
            τ = t - self.n + 1
            if τ >= 0:
                G = sum([self.SARs[i%(self.n+1)].R * self.γ**(i-τ-1) for i in range(τ+1, min(τ+self.n, T)+1)])
                if τ + self.n < T:
                    G += self.Q[self.SARs[(τ+self.n)%(self.n+1)].SA()] * self.γ**self.n
                self.Q[self.SARs[τ%(self.n+1)].SA()] += self.α * (G - self.Q[self.SARs[τ%(self.n+1)].SA()])
                # Update Π:
                self.generateSoftPolicy()
            t+=1

    """ Algo 3: """
    def nStepOffPolicySARSA_Control_episode(self):
        # make a behaviour policy:
        b = self.generateSoftPolicy(0.3)

        def sampleSoftPolicy(state):# choose an action over prob dist Π
            pmf = {a: b[(state,a)] for a in self.env.actions}
            return np.random.choice(list(pmf.keys()), p=list(pmf.values()))

        # Randomly select starting state:
        self.SARs[0].S = self.env.states[np.random.randint(len(self.env.states)-1)]
        self.SARs[0].A = sampleSoftPolicy(self.SARs[0].S)
        state  = self.SARs[0].S
        action = self.SARs[0].A
        T = np.inf
        t, τ = 0, 0
        while τ != T-1:
            if t < T:
                # state = self.SARs[t%(self.n+1)].S
                # action = self.SARs[t%(self.n+1)].A
                # take action, store resultant state and reward at next timestep.
                next_state, reward = self.env.step(state, action)

                self.SARs[(t+1)%(self.n+1)].S = next_state
                self.SARs[(t+1)%(self.n+1)].R = reward
                # if this next state is terminal:T = t+1 else: choose next action from policy
                if next_state in self.env.goals:#self.env.deterministicπEpisodeStillGoing(self.SARs[t%(self.n+1)].S, self.SARs[t%(self.n+1)].A)==False: #state!=self.env.goal
                    T = t+1
                else:
                    # self.SARs[(t+1)%(self.n+1)].A = self.sampleΠ(self.SARs[(t+1)%(self.n+1)].S)
                    state = next_state
                    action = sampleSoftPolicy(state)

            τ = t - self.n + 1
            if τ >= 0:
                ρ = np.prod( [ self.Π[self.SARs[i%(self.n+1)].SA]/self.b[self.SARs[i%(self.n+1)].SA] for i in range(τ+1, min(τ+self.n-1, T-1)+1)] )
                G = sum([self.SARs[i%(self.n+1)].R * self.γ**(i-τ-1) for i in range(τ+1, min(τ+self.n, T)+1)])
                if τ + self.n < T:
                    G += self.Q[self.SARs[(τ+self.n)%(self.n+1)].SA()] * self.γ**self.n
                self.Q[self.SARs[τ%(self.n+1)].SA()] += ρ * self.α * (G - self.Q[self.SARs[τ%(self.n+1)].SA()])
                # Update Π:
                self.generateSoftPolicy()
            t+=1

    """ Algo 4: """
    def nStepOnPolicyTreeBackupSARSA_Control_episode(self):
        # Randomly select starting state:
        self.SARs[0].S = self.env.states[np.random.randint(len(self.env.states)-1)]
        self.SARs[0].A = self.sampleΠ(self.SARs[0].S)
        state  = self.SARs[0].S
        action = self.SARs[0].A
        T = np.inf
        t, τ = 0, 0
        while τ != T-1:
            if t < T:
                # state = self.SARs[t%(self.n+1)].S
                # action = self.SARs[t%(self.n+1)].A
                # take action, store resultant state and reward at next timestep.
                next_state, reward = self.env.step(state, action)

                self.SARs[(t+1)%(self.n+1)].S = next_state
                self.SARs[(t+1)%(self.n+1)].R = reward
                # if this next state is terminal:T = t+1 else: choose next action from policy
                if next_state in self.env.goals:#self.env.deterministicπEpisodeStillGoing(self.SARs[t%(self.n+1)].S, self.SARs[t%(self.n+1)].A)==False: #state!=self.env.goal
                    T = t+1
                else:
                    action = self.sampleΠ(self.SARs[(t+1)%(self.n+1)].S)
                    self.SARs[(t+1)%(self.n+1)].A = action
            τ = t - self.n + 1
            if τ >= 0:
                if t+1 >= T:
                    G = self.SARs[T%(self.n+1)].R
                else:
                    G = self.SARs[(t+1)%(self.n+1)].R + self.γ * sum([ self.Q[(self.SARs[(t+1)%(self.n+1)].S, a)] * self.Π[(self.SARs[(t+1)%(self.n+1)].S, a)] for a in self.env.actions ])
                for k in reversed(range(τ+1, min(t, T-1)+1)):
                    s_k, a_k, r_k = self.SARs[k%(self.n+1)].S, self.SARs[k%(self.n+1)].A, self.SARs[k%(self.n+1)].R
                    G = r_k + self.γ * (self.Π[(s_k, a_k)] * G + sum([ self.Q[(s_k, a)] * self.Π[(s_k, a)]  for a in self.env.actions if a!=a_k ]) )
                self.Q[self.SARs[τ%(self.n+1)].SA()] += self.α * (G - self.Q[self.SARs[τ%(self.n+1)].SA()])
                # Update Π:
                self.generateSoftPolicy()
            t+=1

    """ Algo 5: """
    def nStepOffPolicyTreeBackupSARSA_Control_episode(self):
        # make a behaviour policy:
        b = self.generateSoftPolicy(0.3)

        def sampleSoftPolicy(state):# choose an action over prob dist Π
            pmf = {a: b[(state,a)] for a in self.env.actions}
            return np.random.choice(list(pmf.keys()), p=list(pmf.values()))

        # Randomly select starting state:
        self.SARs[0].S = self.env.states[np.random.randint(len(self.env.states)-1)]
        self.SARs[0].A = sampleSoftPolicy(self.SARs[0].S)
        state = self.SARs[0].S
        action = self.SARs[0].A
        T = np.inf
        t, τ = 0, 0
        while τ != T-1:
            if t < T:
                # state = self.SARs[t%(self.n+1)].S
                # action = self.SARs[t%(self.n+1)].A
                # take action, store resultant state and reward at next timestep.
                next_state, reward = self.env.step(state, action)

                self.SARs[(t+1)%(self.n+1)].S = next_state
                self.SARs[(t+1)%(self.n+1)].R = reward
                # if this next state is terminal:T = t+1 else: choose next action from policy
                if next_state in self.env.goals:#self.env.deterministicπEpisodeStillGoing(self.SARs[t%(self.n+1)].S, self.SARs[t%(self.n+1)].A)==False: #state!=self.env.goal
                    T = t+1
                else:
                    # self.SARs[(t+1)%(self.n+1)].A = self.sampleΠ(self.SARs[(t+1)%(self.n+1)].S)
                    state = next_state
                    action = sampleSoftPolicy(state)
                    self.SARs[(t+1)%(self.n+1)].A = action
                    # σ[(t+1)%(self.n+1)] =

            τ = t - self.n + 1
            if τ >= 0:
                ρ = np.prod( [ self.Π[self.SARs[i%(self.n+1)].SA]/self.b[self.SARs[i%(self.n+1)].SA] for i in range(τ+1, min(τ+self.n-1, T-1)+1)] )
                G = sum([self.SARs[i%(self.n+1)].R * self.γ**(i-τ-1) for i in range(τ+1, min(τ+self.n, T)+1)])
                if τ + self.n < T:
                    G += self.Q[self.SARs[(τ+self.n)%(self.n+1)].SA()] * self.γ**self.n
                self.Q[self.SARs[τ%(self.n+1)].SA()] += ρ * self.α * (G - self.Q[self.SARs[τ%(self.n+1)].SA()])
                # Update Π:
                self.generateSoftPolicy()
            t+=1
