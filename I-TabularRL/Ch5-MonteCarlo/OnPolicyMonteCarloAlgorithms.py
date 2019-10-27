from modules import * #just numpy, grid environment and files in the folder gif_printer

class Node():
    def __init__(self):
        self.visits=0
        self.q=0

class SAR():
    def __init__(self, s, a, r):
        self.s = s
        self.a = a
        self.r = r

class Agent():
    def __init__(self, env, γ=0.9):
        self.γ = γ
        self.env = env
        self.π = {s: np.random.choice(env.actions) for s in env.states}
        self.V = np.zeros((env.w,env.h))
        self.returns = {s:[] for s in env.states}
        self.Q = {(s,a): 0 for s in env.states for a in env.actions}
        self.Returns = {(s,a): Node() for s in env.states for a in env.actions}
        self.Π = {(s,a): 1/len(env.actions) for s in env.states for a in env.actions}

    def generateEpisode(self):
        """ Generate an episode following π:
        episode = [(S0, A0, R1), (S1, A1, R2),...,(S_T-1 , A_T-1, R_T)
        """
        state = self.env.states[np.random.choice(range(len(self.env.states)))]
        episode = []
        while True:
            action = self.π[state]
            next_state, reward = self.env.step(state, action)
            episode.append( SAR(state, action, reward) )
            state = next_state
            if self.env.deterministicπEpisodeStillGoing(state, action)==False: break
        return episode

    def firstVisitMC_V(self,iterations_to_record):
        iteration = 0
        while True:
            # Generate an episode:
            episode = self.generateEpisode()

            # print(episode)
            print(iteration)
            v = np.copy(self.V)

            # Update V
            G = 0
            # for (s,a,r) in reversed(episode):
            for i in reversed(range(len(episode))):
                s,r = episode[i].s, episode[i].r
                G = self.γ*G + r
                if s not in [sar.s for sar in episode[:i]]:# if this is the first occurence
                    self.returns[s].append(G)
                    self.V[s] = np.mean(self.returns[s])

            iteration += 1

            if iteration in iterations_to_record:
                record(iteration, "On-policy MC with state values, iteration: ", self.V, self.π, self.env.w, self.env.h, self.env.costs)
            if iteration>20000 or np.sum(np.abs(self.V - v)) < 1e-2: break

    def ESEpisode(self, starting_state, starting_action):
        """ Generate an episode starting with a given state and action and then following π:
        episode = [(state, action, R1), (S1, A1, R2),...,(S_T-1 , A_T-1, R_T)
        """
        episode = []
        state, action = starting_state, starting_action
        while True:
            next_state, reward = self.env.step(state, action)
            episode.append( SAR(state, action, reward) )
            state = next_state
            action = self.π[state]
            if self.env.deterministicπEpisodeStillGoing(state, action)==False: break
        return episode

    def firstVisitMC_Q(self, iterations_to_record):
        iteration = 0
        while True:
            # Generate an episode using exploring starts:
            starting_state, starting_action = self.env.states[np.random.randint(len(self.env.states))], self.env.actions[np.random.randint(len(self.env.actions))]

            episode = self.ESEpisode(starting_state, starting_action)


            # print(episode)
            print(iteration)
            q = self.Q.copy()

            # Update Q
            G = 0
            for i in reversed(range(len(episode))):
                s,a,r = episode[i].s, episode[i].a, episode[i].r
                G = self.γ*G + r
                if (s,a) not in [(sar.s,sar.a) for sar in episode[:i]]:# if this is the first occurence
                    # self.Returns[(s,a)].append(G)
                    # self.Q[(s,a)] = np.mean(self.Returns[(s,a)])
                    self.Returns[(s,a)].visits += 1
                    n = self.Returns[(s,a)].visits
                    self.Returns[(s,a)].q = G/(n+1) + self.Returns[(s,a)].q*n/(n+1)
                    self.Q[(s,a)] = self.Returns[(s,a)].q
                    # Update π
                    # action_Q = {ac: self.Q[(s,ac)] for ac in self.env.actions}
                    # self.π[s] = max(action_Q, key=action_Q.get)
                    self.π[s] = self.maxAction(s)
            # for (s,a) in episode:
            #     self.Returns[(s,a)].visits += 1
            #     self.Returns[(s,a)].q = reward * 1/(self.Returns[(s,a)].visits + 1) + self.Returns[(s,a)].q* self.Returns[(s,a)].visits/(self.Returns[(s,a)].visits + 1)
            #     self.Q[(s,a)] = self.Returns[(s,a)].q

            iteration += 1

            if iteration in iterations_to_record:
                for s in self.env.states: self.V[s] = sum([self.Q[(s,a)] for a in self.env.actions])
                record(iteration, "On-policy MC with state values, iteration: ", self.V, self.π, self.env.w, self.env.h, self.env.costs)
            if iteration>20000:# or sum([q[(s,a)]-self.Q[(s,a)] for s in self.env.states for a in self.env.actions]) < 1e1:
                for s in self.env.states: self.V[s] = sum([self.Q[(s,a)] for a in self.env.actions])
                record(iteration, "On-policy MC with state values, iteration: ", self.V, self.π, self.env.w, self.env.h, self.env.costs)
                break

    def episodeΠ(self):
        """ Generate an episode following Π:
        episode = [(state, A0, R1), (S1, A1, R2),...,(S_T-1 , A_T-1, R_T)
        """
        state= self.env.states[np.random.randint(len(self.env.states))]
        episode = []

        def sampleΠ(state):# choose an action over prob dist Π
            pmf = {a: self.Π[(state,a)] for a in self.env.actions}
            return np.random.choice(list(pmf.keys()), p=list(pmf.values()))

        while True:
            action = sampleΠ(state)
            next_state, reward = self.env.step(state, action)
            episode.append( SAR(state, action, reward) )
            if self.env.deterministicπEpisodeStillGoing(state, action)==False: break#state!=self.env.goal
            state = next_state
        return episode

    def updatePolicies(self, s):
        # action_Q = {ac: self.Q[(s,ac)] for ac in self.env.actions}
        # a_greedy = max(action_Q, key=action_Q.get)
        self.π[s] = self.maxAction(s)
        for a in self.env.actions:
            if a == self.π[s]:
                self.Π[(s,a)] = 1 - ε + ε/len(self.env.actions)
            else:
                self.Π[(s,a)] = ε/len(self.env.actions)

    def maxAction(self, s):# returns the action with the highest Q-value
        Qs=[(action, self.Q[(s,action)]) for action in self.actions]
        mx = max(Qs, key=lambda x:x[1])[1]
        mxlt = [a for (a,q) in Qs if q==mx]
        return mxlt[np.random.randint(len(mxlt))]

    def firstVisitMC_onPolicy_control_εSoft(self, iterations_to_record, ε):
        iteration = 0
        while True:
            # Generate an episode using exploring starts:
            episode = self.episodeΠ()

            # print(episode)
            print(iteration)
            q = self.Q.copy()

            # Update Q
            G = 0
            for i in reversed(range(len(episode))):
                s,a,r = episode[i].s, episode[i].a, episode[i].r
                G = self.γ*G + r
                if (s,a) not in [(sar.s,sar.a) for sar in episode[:i]]:# if this is the first occurence
                    self.Returns[(s,a)].visits += 1
                    n = self.Returns[(s,a)].visits
                    self.Returns[(s,a)].q = G/(n+1) + self.Returns[(s,a)].q*n/(n+1)
                    self.Q[(s,a)] = self.Returns[(s,a)].q
                    # Update π, Π
                    self.updatePolicies(s)

            iteration += 1
            # record(iteration, self.V, self.π, self.env.w, self.env.h, self.env.costs)
            # print(self.V - v)
            # print(self.V)
            if iteration in iterations_to_record:
                for s in self.env.states: self.V[s] = sum([self.Q[(s,a)] for a in self.env.actions])
                for s in self.env.states:
                    x = {a: self.Π[(s,a)] for a in self.env.actions}
                    self.π[s] = max(x, key=x.get)
                record(iteration, "On-policy MC with state values, iteration: ", self.V, self.π, self.env.w, self.env.h, self.env.costs)
            if iteration>20000:# or sum([q[(s,a)]-self.Q[(s,a)] for s in self.env.states for a in self.env.actions]) < 1e1:
                break

if __name__=="__main__":
    iterations_to_record = list( np.concatenate([np.arange(1,3,1),np.arange(10,100,10),np.arange(500,1000,100),np.arange(1000,20000,1000)]) )

    env = GridWorld()
    agent = Agent(env)
    # agent.firstVisitMC_Q(iterations_to_record)
    agent.firstVisitMC_onPolicy_control_εSoft(iterations_to_record, 0.1)

    makeGIF('../gif_printer/temp-plots', 'gifs/MonteCarlo', 0.5, 12)# framerate=0.25s, 12 repeats at the end
