from modules import * #just numpy, grid environment and files in the folder gif_printer

"""
Off-policy is more complicated than on-policy; we now use two policies:
target policy (learnt about, becomes the optimal policy)
behaviour policy (generates behaviour)
        (we are learning from data 'off' the target policy)
"""
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
        self.Q = {(s,a): 0 for s in env.states for a in env.actions}
        self.Returns = {(s,a): Node() for s in env.states for a in env.actions}
        self.Π = {(s,a): 1/len(env.actions) for s in env.states for a in env.actions}
        self.C = {(s,a): 0 for s in env.states for a in env.actions}
        self.π = {s:np.random.choice(env.actions) for s in env.states}

    def episodeSoftPolicy(self, b, start=None):
        """ Generate an episode following Π:
        episode = [(state, A0, R1), (S1, A1, R2),...,(S_T-1 , A_T-1, R_T)
        """
        state = self.env.states[np.random.randint(len(self.env.states))] if start==None else start
        episode = []

        def sampleSoftPolicy(state):# choose an action over prob dist Π
            pmf = {a: b[(state,a)] for a in self.env.actions}
            return np.random.choice(list(pmf.keys()), p=list(pmf.values()))

        while True:
            action = sampleSoftPolicy(state)
            next_state, reward = self.env.step(state, action)
            episode.append( SAR(state, action, reward) )
            if self.env.deterministicπEpisodeStillGoing(state, action)==False: break#state!=self.env.goal
            state = next_state
        return episode

    def generateSoftPolicy(self, ε):
        Π = {}
        for s in self.env.states:
            for a in self.env.actions:
                if a == self.π[s]:
                    Π[(s,a)] = 1 - ε + ε/len(self.env.actions)
                else:
                    Π[(s,a)] = ε/len(self.env.actions)
        return Π

    def maxAction(self, s):# returns the action with the highest Q-value
        Qs=[(action, self.Q[(s,action)]) for action in self.env.actions]
        mx = max(Qs, key=lambda x:x[1])[1]
        mxlt = [a for (a,q) in Qs if q==mx]
        return mxlt[np.random.randint(len(mxlt))]

    def off_policy_MC_Q(self, iterations_to_record):
        iteration = 0
        while True:
            # make a behaviour policy:
            b = self.generateSoftPolicy(0.3)
            # Generate an episode following b:
            episode = self.episodeSoftPolicy(b)

            # print(episode)
            print(iteration)
            q = self.Q.copy()

            # Update Q
            G = 0
            W = 1
            for i in reversed(range(len(episode))):
                # prediction...
                s,a,r = episode[i].s, episode[i].a, episode[i].r
                G = self.γ*G + r
                self.C[(s, a)] += W
                self.Q[(s, a)] += (G - self.Q[(s, a)]) * W / self.C[(s, a)]
                # W *= self.Π[(s, a)] / b[(s, a)]
                # control...
                # x = {ac: self.Q[(s,ac)] for ac in self.env.actions}
                # self.π[s] = max(x, key=x.get)
                self.π[s] = self.maxAction(s)
                if a != self.π[s]: break
                W /= b[(s, a)]

            iteration += 1
            # record(iteration, self.V, self.π, self.env.w, self.env.h, self.env.costs)
            # print(self.V - v)
            # print(self.V)
            if iteration in iterations_to_record:
                V = {s: sum([self.Q[(s,a)] for a in self.env.actions]) for s in self.env.states}
                record(iteration, "Policy iteration with state values, iteration: ", V, self.π, self.env.w, self.env.h, self.env.costs)
            if iteration>20000: break# or sum([q[(s,a)]-self.Q[(s,a)] for s in self.env.states for a in self.env.actions]) < 1e1:
            if W==0: break

if __name__=="__main__":
    iterations_to_record = list( np.concatenate([np.arange(1,3,1),np.arange(10,100,10),np.arange(500,1000,100),np.arange(1000,20000,1000)]) )

    env = GridWorld()
    agent = Agent(env)
    agent.off_policy_MC_Q(iterations_to_record)

    makeGIF('../gif_printer/temp-plots', 'gifs/MonteCarloOffPolicy', 0.5, 12)# framerate=0.25s, 12 repeats at the end
