from TDLearning import *

class SR():
    def __init__(self, s, r):
        self.s = s
        self.r = r

class MRP():
    def __init__(self, env, α=0.1, γ=0.9):
        self.env = env
        self.α = α
        self.γ = γ
        self.V = np.ones(7)*0.5
        self.V[env.goals] = 0
        self.returns = np.zeros(8)
        self.visits = np.ones(8, dtype=int)

    def run_episode(self):
        state = self.env.start
        episode = []
        while True:
            next_state, reward = self.env.step(state)
            episode.append( SR(state, reward) )
            if state in self.env.goals: break
            state = next_state
        return episode

    def TD0VPredictionEpisode(self):
        state = self.env.start
        while True:
            next_state, reward = self.env.step(state)
            self.V[state] += self.α*(reward + self.γ*self.V[next_state] - self.V[state])
            state = next_state
            if state in self.env.goals: break

    def MCVPredictionEpisode(self):
        episode = self.run_episode()

        # Update V
        G = 0
        for i in reversed(range(len(episode))):
            s,r = episode[i].s, episode[i].r
            G = self.γ * G + r
            if s not in [sr.s for sr in episode[:i]]:# if this is the first occurence
                self.returns[s] = G/(self.visits[s]+1) + self.returns[s]*self.visits[s]/(self.visits[s]+1)
                self.V[s] += self.α * (self.returns[s] - self.V[s])
                self.visits[s] += 1

    def batch_TD_update(self, episodes, α):
        while True:# keep giving episodes until V converges
            updates = np.zeros(7)
            for episode in episodes:
                for i in range(len(episode) - 1):
                    updates[episode[i].s] += episode[i].r + self.V[episode[i + 1].s] - self.V[episode[i].s]
            updates *= α
            if np.sum(np.abs(updates)) < 1e-3: break
            self.V += updates

    def batch_MC_update(self, episodes, α):
        while True:# keep giving episodes until V converges
            updates = np.zeros(7)
            for episode in episodes:
                ep_rew = sum([episode[j].r for j in range(len(episode))])
                for i in range(len(episode) - 1):
                    updates[episode[i].s] += ep_rew - self.V[episode[i].s]
            updates *= α
            if np.sum(np.abs(updates)) < 1e-3: break
            self.V += updates

def example_6_2a():
    env = Walk()
    agent = MRP(env, α=0.1, γ=1.0)
    its = [0,1,10,100]
    labels = [str(l) for l in its]+["True values"]
    data = {}
    for i in range(101):
        if str(i) in labels: data[str(i)] = (list(range(5)), list(agent.V[1:-1]))
        agent.TD0VPredictionEpisode()

    data.update({"True values":(list(range(len(agent.env.true_V)-2)), list(agent.env.true_V[1:-1]))})
    multipleCurvesPlot(data, "Estimated value", "State", "ylabel", "example_6_2a", lgplc='lowers right',
                        w=6.0, ylims=[0,1], labels=labels, xticks=([0,1,2,3,4],['A','B','C','D','E']), xlog=None)

def example_6_2b():
    env = Walk()
    data = {}

    episodes = np.arange(101)
    αs     = [0.05, 0.1, 0.15, 0.01, 0.02, 0.03, 0.04]
    runs   = 100
    labels = ['TD, α = '+str(α) if αi < 3 else 'MC, α = '+str(α) for αi, α in enumerate(αs)]
    rms    = np.zeros((len(αs), len(episodes)))

    for _ in range(runs):
        for αi, α in enumerate(αs):
            agent = MRP(env, α=α, γ=1.0)
            if αi < 3:
                for ep in episodes:
                    agent.TD0VPredictionEpisode()
                    rms[αi, ep] += np.sqrt(np.mean(np.power(agent.V[1:-1] - env.true_V[1:-1], 2))) / runs
            else:
                for ep in episodes:
                    agent.MCVPredictionEpisode()
                    rms[αi, ep] += np.sqrt(np.mean(np.power(agent.V[1:-1] - env.true_V[1:-1], 2))) / runs
    data = {l:(list(range(len(rms[li]))), list(rms[li])) for li, l in enumerate(labels)}
    
    data.update({"True values":(list(range(len(agent.env.true_V)-2)), list(agent.env.true_V[1:-1]))})
    multipleCurvesPlot(data, "Empirical RMS error, averaged over states", "Walks/Episodes", "RMS Error in V", "example_6_2b", lgplc='upper right',
                        w=6.0, ylims=[0,0.25], labels=labels)

def figure_6_2():
    env = Walk()
    runs = 100
    epis = 50
    α=0.005
    total_errors = np.zeros((2, epis))
    for (i, fn) in [(0, 'batch_TD_update'), (1, 'batch_MC_update')]:
        for _ in tqdm(range(0, runs)):
            errors   = []
            episodes = []
            agent = MRP(env, α=α, γ=1.0)
            for ep in range(epis):
                episode = agent.run_episode()
                episodes.append(episode)

                object = getattr(agent, fn)
                object(episodes, α)# calling batch_TD_update and batch_MC_update

                errors.append(np.sqrt(np.sum(np.power(agent.V - env.true_V, 2)) / 5.0))
            total_errors[i] += np.array(errors)
        total_errors[i] /= runs

    data = {'TD':(list(range(epis)), list(total_errors[0])),
            'MC':(list(range(epis)), list(total_errors[1]))}

    multipleCurvesPlot(data, "Batch Training", "Walks/Episodes", "RMS Error averaged over states", 
                       "figure_6_2", lgplc='upper right', w=6.0, ylims=[0,0.25])

def figure_6_5():
    env = MDP_Norm()

    runs, epis = 10000, 300
    left_actions = np.zeros((2, epis))
    for (i, fn) in [(0, 'QLearningEpisode'), (1, 'doubleQLearningEpisode')]:
        for _ in tqdm(range(runs)):
            agent = MaxBiasExample(env, γ=1.0, α=0.1, ε=0.1, sm="εGreedy")
            for ep in range(epis):
                object = getattr(agent, fn)
                actions = object(env.start)# calling QLearningEpisode and doubleQLearningEpisode

                proportion = actions.count('left')/len(actions)
                left_actions[i, ep] += proportion/runs

    data = {'Q-Learning':(list(range(epis)),list(left_actions[0])), 
            'Double Q-Learning':(list(range(epis)),list(left_actions[1]))}

    multipleCurvesPlot(data, "Maximising Bias", "Episodes", "Proportion of left actions", 
                       "figure_6_5", lgplc='upper right', w=6.0, ylims=[0,1])

def my_gridworld():
    iterations_to_record = list( np.concatenate([np.arange(1,3,1),np.arange(10,100,10),np.arange(500,1000,100),np.arange(1000,20000,1000)]) )

    env = GridWorld()
    agent = Agent(env)

    agent.sarsa(iterations_to_record)

    makeGIF('../gif_printer/temp-plots', 'gifs/Sarsa', 0.5, 12)# framerate=0.25s, 12 repeats at the end


if __name__=="__main__":
    example_6_2a()
    example_6_2b()
    figure_6_2()
    figure_6_5()

    my_gridworld()
    
