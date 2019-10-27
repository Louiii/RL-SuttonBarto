from modules import *

class MaxBiasExample():
    def __init__(self, env, γ=1.0, α=0.1, ε=0.1, sm="εGreedy", vinit=None):
        self.env = env
        self.γ = γ
        self.α = α
        self.ε = ε
        self.selectionMethod = sm

        self.V = {s: 0 for s in env.states}
        self.Q  = {(s,a): 0 for s in env.states for a in env.actions[s]}
        self.Q2 = {(s,a): 0 for s in env.states for a in env.actions[s]}
        self.π = {s: np.random.choice(env.actions[s]) for s in env.states}

    def maxAction(self, s, double=False, second=False):# returns the action with the highest Q-value
        if double==True:
            Qs=[(action, self.Q[(s,action)]+self.Q2[(s,action)]) for action in self.env.actions[s]]
        elif second==True:
            Qs=[(action, self.Q2[(s,action)]) for action in self.env.actions[s]]
        else:
            Qs=[(action, self.Q[(s,action)]) for action in self.env.actions[s]]
        mx = max(Qs, key=lambda x:x[1])[1]
        mxlt = [a for (a,q) in Qs if q==mx]
        return mxlt[np.random.randint(len(mxlt))]

    def updateπ(self, double=False):
        for s in self.env.states: self.π[s] = self.maxAction(s, double)

    def policy(self, state, double=False):
        if self.selectionMethod=="Random":
            return np.random.choice(self.env.actions[state])
        elif self.selectionMethod=="εGreedy":
            if self.ε < np.random.rand():
                return self.maxAction(state, double=double)
            return np.random.choice(self.env.actions[state])

    def QLearningEpisode(self, start=None):
        """Off-policy TD control for estimating π"""
        state = start if start is not None else self.env.states[np.random.choice(range(len(self.env.states)))]
        actions = []
        while True:
            action = self.policy(state)
            next_state, reward = self.env.step(state, action)

            self.Q[(state,action)] += self.α*(reward + self.γ*self.Q[(next_state, self.maxAction(next_state))] - self.Q[(state, action)])
            # if self.env.deterministicπEpisodeStillGoing(state, action)==False: break#state!=self.env.goal
            if state=='A':actions.append(action)
            state = next_state
            if state in self.env.goals: break
        return actions

    def doubleQLearningEpisode(self, start=None):
        """Off-policy TD control for estimating π"""
        actions = []
        state = start if start is not None else self.env.states[np.random.choice(range(len(self.env.states)))]
        while True:
            action = self.policy(state, double=True)
            next_state, r = self.env.step(state, action)

            if np.random.randint(2) == 0:
                self.Q[(state,action)] += self.α*(r + self.γ*self.Q2[(next_state, self.maxAction(next_state))] - self.Q[(state, action)])
            else:
                self.Q2[(state,action)] += self.α*(r + self.γ*self.Q[(next_state, self.maxAction(next_state, double=False, second=True))] - self.Q2[(state, action)])

            # if self.env.deterministicπEpisodeStillGoing(state, action)==False: break#state!=self.env.goal
            if state=='A':actions.append(action)
            state = next_state
            if state in self.env.goals: break
        return actions

class Agent():
    def __init__(self, env, γ=0.9, α=0.5, ε=0.1, sm="εGreedy", vinit=None):
        self.env = env
        self.γ = γ
        self.α = α
        self.ε = ε
        self.selectionMethod = sm

        self.π = {s: np.random.choice(env.actions) for s in env.states}
        self.V = {s: np.random.randn() for s in env.states} if vinit==None else {s:0 for s in env.states}
        self.V[env.goal] = 0
        self.Q = {(s,a): np.random.randn() for s in env.states for a in env.actions}
        for a in env.actions: self.Q[(env.goal,a)] = 0
        self.Q2 = {(s,a): np.random.randn() for s in env.states for a in env.actions}
        for a in env.actions: self.Q2[(env.goal,a)] = 0

    def maxAction(self, s, double=False, second=False):# returns the action with the highest Q-value
        if double==True:
            Qs=[(action, self.Q[(s,action)]+self.Q2[(s,action)]) for action in self.actions]
        elif second==True:
            Qs=[(action, self.Q2[(s,action)]) for action in self.actions]
        else:
            Qs=[(action, self.Q[(s,action)]) for action in self.actions]
        mx = max(Qs, key=lambda x:x[1])[1]
        mxlt = [a for (a,q) in Qs if q==mx]
        return mxlt[np.random.randint(len(mxlt))]

    def updateπ(self, double=False):
        for s in self.env.states: self.π[s] = self.maxAction(s, double)

    def policy(self, state, double=False):
        if self.selectionMethod=="Random":
            return np.random.choice(self.env.actions)
        elif self.selectionMethod=="εGreedy":
            if self.ε < np.random.rand():
                return self.maxAction(state, double=double)
            return np.random.choice(self.env.actions)

    def TD0VPredictionEpisode(self, start=None):
        # while True:
        state = start if start is not None else self.env.states[np.random.choice(range(len(self.env.states)))]
        # episode:
        while True:
            action = self.policy(state)

            next_state, reward = self.step(state, action)
            self.V[state] += self.α*(reward + self.γ*self.V[next_state] - self.V[state])
            if state in self.env.goals:#self.env.deterministicπEpisodeStillGoing(state, action)==False: 
                break#state!=self.env.goal
            state = next_state

    def sarsa(self, iterations_to_record):
        """ on-policy TD control for estimating Q """
        iteration=0
        while True:
            iteration+=1

            state = self.env.states[np.random.choice(range(len(self.env.states)))]
            action = self.policy(state)
            # episode:
            while True:
                next_state, reward = self.env.step(state, action)

                next_action = self.policy(next_state)

                self.Q[(state,action)] += self.α*(reward + self.γ*self.Q[(next_state, next_action)] - self.Q[(state, action)])
                if self.env.deterministicπEpisodeStillGoing(state, action)==False: break#state!=self.env.goal

                state, action = next_state, next_action

            if iteration in iterations_to_record:
                # self.π = {s: max([(a,self.Q[(s,a)]) for a in self.env.actions], key=lambda x:x[1])[0] for s in self.env.states}
                self.updateπ()
                for s in self.env.states: self.V[s] = sum([self.Q[(s,a)] for a in self.env.actions])
                record(iteration, "Policy iteration with state values, iteration: ", self.V, self.π, self.env.w, self.env.h, self.env.costs)
            if iteration>20000:# or sum([q[(s,a)]-self.Q[(s,a)] for s in self.env.states for a in self.env.actions]) < 1e1:
                break

    def QLearning(self, iterations_to_record):
        """Off-policy TD control for estimating π"""
        iteration=0
        while True:
            iteration+=1

            state = self.env.states[np.random.choice(range(len(self.env.states)))]
            # episode:
            while True:
                action = self.policy(state)
                next_state, reward = self.env.step(state, action)

                self.Q[(state,action)] += self.α*(reward + self.γ*self.Q[(next_state, self.maxAction(next_state))] - self.Q[(state, action)])
                if self.env.deterministicπEpisodeStillGoing(state, action)==False: break#state!=self.env.goal

                state = next_state


            if iteration in iterations_to_record:
                self.updateπ()
                for s in self.env.states: self.V[s] = sum([self.Q[(s,a)] for a in self.env.actions])
                record(iteration, "Policy iteration with state values, iteration: ", self.V, self.π, self.env.w, self.env.h, self.env.costs)
            if iteration>20000:# or sum([q[(s,a)]-self.Q[(s,a)] for s in self.env.states for a in self.env.actions]) < 1e1:
                break

    def doubleQLearning(self, iterations_to_record):
        """Off-policy TD control for estimating π"""
        iteration=0
        while True:
            iteration+=1

            state = self.env.states[np.random.choice(range(len(self.env.states)))]
            # episode:
            while True:
                action = self.policy(state, double=True)
                next_state, r = self.env.step(state, action)

                if np.random.randint(2) == 0:
                    self.Q[(state,action)] += self.α*(r + self.γ*self.Q2[(next_state, self.maxAction(next_state))] - self.Q[(state, action)])
                else:
                    self.Q2[(state,action)] += self.α*(r + self.γ*self.Q[(next_state, self.maxAction(next_state, double=False, second=True))] - self.Q2[(state, action)])

                if self.env.deterministicπEpisodeStillGoing(state, action)==False: break#state!=self.env.goal

                state = next_state


            if iteration in iterations_to_record:
                for s in self.env.states: self.π[s] = max([(a,self.Q[(s,a)]+self.Q2[(s,a)]) for a in self.env.actions], key=lambda x:x[1])[0]
                for s in self.env.states: self.V[s] = sum([self.Q[(s,a)]+self.Q2[(s,a)] for a in self.env.actions])
                record(iteration, "Policy iteration with state values, iteration: ", self.V, self.π, self.env.w, self.env.h, self.env.costs)
            if iteration>20000:# or sum([q[(s,a)]-self.Q[(s,a)] for s in self.env.states for a in self.env.actions]) < 1e1:
                break
