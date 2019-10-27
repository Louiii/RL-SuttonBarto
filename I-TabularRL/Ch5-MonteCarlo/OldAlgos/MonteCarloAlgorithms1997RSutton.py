from modules import * #just numpy, grid environment and files in the folder gif_printer

class Node():
    def __init__(self):
        self.visits=0
        self.q=0

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
        state = self.env.states[np.random.choice(range(len(self.env.states)))]
        episode = [state]
        totalReward=0
        while self.env.deterministicπEpisodeStillGoing(state, self.π[state]):
            state, r = self.env.step(state, self.π[state])
            episode.append(state)
            totalReward += r
        return episode, totalReward

    def firstVisitMC_V(self):
        iteration = 0
        while True:
            # Generate an episode:
            episode, reward = self.generateEpisode()

            # print(episode)
            print(iteration)
            v = np.copy(self.V)

            # Update V
            for s in episode:
                self.returns[s].append(reward)
                self.V[s] = np.mean(self.returns[s])

            iteration += 1
            # record(iteration, self.V, self.π, self.env.w, self.env.h, self.env.costs)
            # print(self.V - v)
            # print(self.V)
            if np.sum(np.abs(self.V - v)) < 1e-2 and iteration > 20000:
                record(iteration, self.V, self.π, self.env.w, self.env.h, self.env.costs)
                break

    def exploringStartsEpisode(self):
        (state, a) = (self.env.states[np.random.randint(len(self.env.states))], self.env.actions[np.random.randint(len(self.env.actions))])
        episode = [(state, a)]
        state, totalReward = self.env.step(state, a)
        episode.append((state, self.π[state]))
        while self.env.deterministicπEpisodeStillGoing(state, self.π[state]):
            state, r = self.env.step(state, self.π[state])
            episode.append((state, self.π[state]))
            totalReward += r
        return episode, totalReward

    def firstVisitMC_Q(self):
        iteration = 0
        while True:
            # Generate an episode using exploring starts:
            episode, reward = self.exploringStartsEpisode()

            # print(episode)
            print(iteration)
            q = self.Q.copy()

            # Update Q
            for (s,a) in episode:
                self.Returns[(s,a)].visits += 1
                self.Returns[(s,a)].q = reward * 1/(self.Returns[(s,a)].visits + 1) + self.Returns[(s,a)].q* self.Returns[(s,a)].visits/(self.Returns[(s,a)].visits + 1)
                self.Q[(s,a)] = self.Returns[(s,a)].q

            # Update π
            for (s,_) in episode:
                action_Q = {a: self.Q[(s,a)] for a in self.env.actions}
                self.π[s] = max(action_Q, key=action_Q.get)

            iteration += 1
            # record(iteration, self.V, self.π, self.env.w, self.env.h, self.env.costs)
            # print(self.V - v)
            # print(self.V)
            if iteration > 100000 and sum([q[(s,a)]-self.Q[(s,a)] for s in self.env.states for a in self.env.actions]) < 1e1:
                for s in self.env.states: self.V[s] = sum([self.Q[(s,a)] for a in self.env.actions])
                record(iteration, self.V, self.π, self.env.w, self.env.h, self.env.costs)
                break

    def ESEpisodeΠ(self):
        # Exploring start:
        (state, action) = (self.env.states[np.random.randint(len(self.env.states))], self.env.actions[np.random.randint(len(self.env.actions))])
        episode = [(state, action)]
        state, totalReward = self.env.step(state, action)

        def sampleΠ(state):# choose an action over prob dist Π
            pmf = {a: self.Π[(state,a)] for a in self.env.actions}
            return np.random.choice(pmf.keys(), p=pmf.values())

        while True:
            action = sampleΠ(state)
            state, r = self.env.step(state, action)
            episode.append((state, action))
            totalReward += r
            if state!=goal: break#self.env.deterministicπEpisodeStillGoing(state, self.π[state])==False
        return episode, totalReward

    def firstVisitMC_onPolicy_control_εSoft(self, ε):
        iteration = 0
        while True:
            # Generate an episode using exploring starts:
            episode, reward = self.ESEpisodeΠ()

            # print(episode)
            print(iteration)
            q = self.Q.copy()

            # Update Q
            for (s,a) in episode:
                self.Returns[(s,a)].visits += 1
                self.Returns[(s,a)].q = reward * 1/(self.Returns[(s,a)].visits + 1) + self.Returns[(s,a)].q* self.Returns[(s,a)].visits/(self.Returns[(s,a)].visits + 1)
                self.Q[(s,a)] = self.Returns[(s,a)].q

            # Update π
            for (s,_) in episode:
                action_Q = {a: self.Q[(s,a)] for a in self.env.actions}
                # self.π[s] = max(action_Q, key=action_Q.get)
                a_greedy = max(action_Q, key=action_Q.get)

                for a in self.env.actions:
                    if a == a_greedy:
                        self.Π[(s,a_greedy)] = 1 - ε + ε/len(self.env.actions)
                    else:
                        self.Π[(s,a)] = ε/len(self.env.actions)

            iteration += 1
            # record(iteration, self.V, self.π, self.env.w, self.env.h, self.env.costs)
            # print(self.V - v)
            # print(self.V)
            if iteration > 1000000 and sum([q[(s,a)]-self.Q[(s,a)] for s in self.env.states for a in self.env.actions]) < 1e1:
                for s in self.env.states: self.V[s] = sum([self.Q[(s,a)] for a in self.env.actions])
                for s in self.env.states:
                    x = {a: self.Π[(s,a)] for a in self.env.actions}
                    self.π[s] = max(x, key=x.get)
                record(iteration, self.V, self.π, self.env.w, self.env.h, self.env.costs)
                break


def record(iteration, V, π, w, h, costs):
    X, Y, u, v = makeUVM(π, w, h)
    fn = "../gif_printer/temp-plots/V"+str(iteration)+".png"
    title = "Policy iteration with state values, iteration: "+str(iteration)
    plotPolicy(V, X, Y, u, v, costs, w, h, show=False, filename=fn, title=title, cbarlbl="Costs")

if __name__=="__main__":
    env = Environment()
    agent = Agent(env)
    agent.firstVisitMC_onPolicy_control_εSoft(0.4)

    makeGIF('../gif_printer/temp-plots', 'gifs/MonteCarlo', 0.5, 12)# framerate=0.25s, 12 repeats at the end
