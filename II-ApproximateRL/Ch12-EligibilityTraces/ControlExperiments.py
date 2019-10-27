from EligibilityTracesControl import *
from CartPoleEligibilitytraces import *
from EligibilityTraces import *

def figure_12_10(simulate=True):
    αs = [0.25*i for i in range(1, 8)]
    λs = np.array([0, 0.68, 0.84, 0.92, 0.96, 0.98, 0.99])
    labels = ['λ = '+str(λ) for λ in λs]
    episodes = 50
    runs = 10# 100 takes about 2 hours
    if simulate:
        steps = np.zeros((len(λs), len(αs)))
        for run in tqdm(range(runs)):
            for λi, λ in enumerate(λs):
                for αi, α in enumerate(αs):
                    car = MountainCar()
                    agent = Agent(car, 1, λ, QFnArgs=[α])
                    for episode in range(episodes):
                        step, _ = agent.sarsaλ()
                        steps[λi, αi] += step / (runs * episodes)
                    # print("\nλ = "+str(λ)+", α = "+str(α)+", steps = "+str(steps[λi, αi]))
        data = {labels[i] : (αs, list(steps[i])) for i in range(len(labels))}
        export_dataset(data, "fig12_10data")# Store the data from the simulation

    # Load the data from a simulation
    data = load_dataset("fig12_10data")
    multipleCurvesPlot(data, "Sarsa(λ) with replacing traces", 'α * number of tilings(8)', 'Mountain Car\nSteps per episode\naveraged over \nfirst '+str(episodes)+' episodes\nand '+str(runs)+'runs',
                    "figure_12_10", w=6.0, h=4.5, labels=labels)#, ylims=[220, max_steps])


def figure_12_11(simulate=True):
    αs = list(np.arange(0.2, 2.2, 0.2))
    labels = ["True online Sarsa(λ)", "Replacing traces Sarsa(λ)",
              "Clearing traces Sarsa(λ)", "Accumulating traces Sarsa(λ)"]
    if simulate:
        episodes = 20
        runs = 15
        data = { label: {αi:[] for αi, α in enumerate(αs)} for label in labels }
        data[labels[-1]] = {αi:[] for αi in range(sum([α < 0.65 for α in αs]))}
        for αi, α in enumerate(αs):# x axis
            for run in tqdm(range(runs)):# repeats
                # init
                trueOnline_agent   = Agent(MountainCar(), 1, λ=0.9, QFnArgs=[α])
                replacingT_agent   = Agent(MountainCar(), 1, λ=0.9, QFnArgs=[α])
                clearing_agent     = Agent(MountainCar(), 1, λ=0.9, QFnArgs=[α])
                accumulating_agent = Agent(MountainCar(), 1, λ=0.9, QFnArgs=[α])
                for episode in range(episodes):# episodes
                    _, reward1 = trueOnline_agent.trueOnline_sarsaλ()
                    _, reward2 = replacingT_agent.sarsaλ()
                    _, reward3 = clearing_agent.sarsaλ(clearing=True)
                    rewards = [reward1, reward2, reward3]
                    if α < 0.65:
                        _, reward4 = accumulating_agent.sarsaλ(replacing_traces=False)
                        rewards.append( reward4 )
                    # update data
                    for i in range(len(rewards)):
                        data[labels[i]][αi].append( rewards[i] )
            for label in labels:
                if label != labels[-1] or α < 0.65:
                    data[label][αi] = np.mean(data[label][αi])
        print(data)
        for label in labels:
            if label == labels[-1]: αs = [0.2, 0.4, 0.6]
            print(αs)
            data[label] = (list(αs), [data[label][αi] for αi, α in enumerate(αs)])
        print(data)
        export_dataset(data, "fig12_11data")
    data = load_dataset("fig12_11data")
    print(data)
    multipleCurvesPlot(data, "", 'α * number of tilings(8)', 'Mountain Car reward per episode\n(averaged over first '+str(episodes)+' episodes and '+str(runs)+' runs)',
                    "figure_12_11", lgplc='lower right', w=6.0, h=4.5, labels=labels, ylims=[-550, -150])
def returnQuartilesErrors(arr):
    arr = arr.flatten()
    q25 = np.percentile(arr, 25)
    q50 = np.percentile(arr, 50)
    q75 = np.percentile(arr, 75)
    return (q50-q25, q50, q75-q50)
    # mn = np.mean(arr)
    # return (mn - min(arr), mn, max(arr) - mn)

def figure_12_14_MountainCar(simulate=True):
    ac_λs = [0, 0.3, 0.5, 0.6, 0.65]
    rp_λs = [0, 0.4, 0.7, 0.8, 0.9, 0.93, 0.97, 1]
    labels = ['accumulating traces', 'replacing traces']
    α = 1.2
    if simulate:
        episodes = 50
        runs = 3
        acsteps = np.zeros((len(ac_λs), runs, episodes))
        rpsteps = np.zeros((len(rp_λs), runs, episodes))
        for run in tqdm(range(runs)):
            for rp in [True, False]:
                for λi, λ in enumerate(ac_λs):
                    agent = Agent(MountainCar(), 1, λ, QFnArgs=[α])
                    for episode in range(episodes):
                        step, _ = agent.sarsaλ(replacing_traces=rp)
                        if rp:
                            rpsteps[λi, run, episode] += step
                        else:
                            acsteps[λi, run, episode] += step
        acsteps = [returnQuartilesErrors(acsteps[λi]) for λi, λ in enumerate(ac_λs)]
        rpsteps = [returnQuartilesErrors(rpsteps[λi]) for λi, λ in enumerate(rp_λs)]
        acsteps = [list(a) for a in list(zip(*acsteps))]
        rpsteps = [list(a) for a in list(zip(*rpsteps))]
        data = {labels[0]:(ac_λs, acsteps[1], acsteps[0], acsteps[2]),
                labels[1]:(rp_λs, rpsteps[1], rpsteps[0], rpsteps[2]), 'α':α}
        export_dataset(data, "fig12_14MCdata")# Store the data from the simulation

    # Load the data from a simulation
    data = load_dataset("fig12_14MCdata")
    α = data['α']
    del data['α']
    data.update({"xlab":"", "ylab":"Steps per episode", "title":"Mountain Car, α = "+str(α), "ylims":[100, 400]})
    return data

def figure_12_14_RandomWalk(simulate=True):
    λs = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.925, 0.95, 0.975, 1.0]
    labels = ['accumulating traces', 'replacing traces']
    αs = [0.85, 0.8, 0.75, 0.55, 0.35, 0.25, 0.2, 0.15, 0.1, 0.05]
    n_episodes = 10
    repeats = 500
    if simulate: run_sim(n_episodes, repeats, λs, αs, labels, "fig12_14RWdata", "semiGradientTDλ_episode")

    # Load the data from a simulation
    data = load_dataset("fig12_14RWdata")
    α = "best α for accumulating traces"#data['α']
    del data['α']
    data.update({"xlab":"", "ylab":"RMS error", "title":"Random Walk, α = "+str(α), "ylims":[0.15, 0.5]})
    return data

def run_sim(n_episodes, repeats, λs, αs, labels, fname, fn):
    errors = np.zeros((len(labels), len(λs), repeats))
    for λi, λ in tqdm(enumerate(λs)):
        α = αs[λi]
        for i in range(repeats):
            for li, label in enumerate(labels):
                agent = predAgent(Walk(), n_weights=21, λ=λ, α=α)
                for _ in range(n_episodes):
                    episode = getattr(agent, fn)
                    episode(trace=label)
                    # compute error:
                    difference = np.array([agent.V.value(s) for s in agent.env.states]) - agent.env.true_V[1:-1]
                    errors[li, λi, i] += np.sqrt(np.mean(np.power(difference, 2)))/n_episodes
    # average_rms = np.mean(errors, 2)/n_episodes# 2 is the axis for repeats
    acsteps = [returnQuartilesErrors(errors[0, λi]) for λi, λ in enumerate(λs)]
    rpsteps = [returnQuartilesErrors(errors[1, λi]) for λi, λ in enumerate(λs)]
    acsteps = [list(a) for a in list(zip(*acsteps))]
    rpsteps = [list(a) for a in list(zip(*rpsteps))]
    data = {labels[0]:(λs, acsteps[1], acsteps[0], acsteps[2]),
            labels[1]:(λs, rpsteps[1], rpsteps[0], rpsteps[2]), 'α':αs}
    export_dataset(data, fname)

def figure_12_14_PuddleWorld():
    n_episodes = 50
    labels = ['replacing traces']
    λs = [0, 0.5, 0.8, 0.9, 0.9, 0.95, 0.97, 0.98, 0.99]
    α = 0.5
    repeats = 50
    rpcosts = np.zeros((len(λs), repeats, n_episodes))
    for λi, λ in tqdm(enumerate(λs)):
        for run in range(repeats):
            for li, label in enumerate(labels):
                agent = Agent(PW(), 1, λ, QFnArgs=[α], pw=True)
                for episode in range(n_episodes):
                    _, episode_reward = agent.sarsaλ(start=agent.env.start)
                    rpcosts[λi, run, episode] = -episode_reward
    rpcosts = [returnQuartilesErrors(rpcosts[λi]) for λi, λ in enumerate(λs)]
    rpcosts = [list(a) for a in list(zip(*rpcosts))]
    data = {"xlab":"λ", "ylab":"Cost per Episode", "title":"PuddleWorld", "ylims":[0, 10],
           labels[0]:(λs, rpcosts[1], rpcosts[0], rpcosts[2])}
    return data


def figure_12_14_CartAndPole(simulate=True):
    # return figure_12_14_PuddleWorld()
    if simulate:
        γ = 1.0
        ε = 0.0
        labels = ['accumulating traces']
        λs = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
        αs = [0.25,0.25,0.25,0.25, 0.33,0.36,0.3]
        max_steps = 1e5
        runs = 5
        errors = np.zeros((len(λs), runs))
        for run in range(runs):# x runs # 2 hours x runs
            for λi, λ in tqdm(enumerate(λs)):# x 7 = 140 mins
                α = αs[λi]
                cartpole = CartPole(α, λ, γ, ε)
                steps = 0
                failures = 0
                while steps < 1e5:#1e5 takes ~ 20 mins
                    if steps%50==0: print("check"+str(steps) + 'failures = '+str(failures)+', predicted = '+str(failures*1e5/(steps+1)))
                    steps += cartpole.sarsaλ(max_steps, replacing_traces=True)
                    failures += 1
                errors[λi][run] = failures
        failures = [returnQuartilesErrors(errors[λi]) for λi, λ in enumerate(λs)]
        failures = [list(a) for a in list(zip(*failures))]
        data = {"xlab":"λ", "ylab":"Failures per 100,000 steps", "title":"Cart & Pole", "ylims":[50, 300],
               labels[0]:(λs, failures[1], failures[0], failures[2])}
        export_dataset(data, "fig12_14CPdata")
    # return load_dataset("fig12_14CPdata")
    data = load_dataset("fig12_14CPdata")
    data["ylims"] = [500, 1500]
    return data

def cartpole_rec_episode():
    from gym import wrappers
    γ = 1.0
    ε = 0.0
    λ = 0.2
    α = 0.25
    max_steps = 1e5
    cartpole = CartPole(α, λ, γ, ε)
    steps = 0
    while steps < max_steps:
        steps += cartpole.sarsaλ(max_steps, replacing_traces=True)
        print("Steps: ", str(steps), ", ", str(100*steps/max_steps)+"%")

    def record(name):
        max_steps = 10000
        env_to_wrap = gym.make('CartPole-v0')
        env = wrappers.Monitor(env_to_wrap, 'videos/'+name, force = True)
        _ = cartpole.render_episode(env, max_steps)
        env.close()
        env_to_wrap.close()

    for i in range(3):
        record("cartpole_episode"+str(i))


# def find_alphas_CartAndPole(simulate=True):
#     λs = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
#     labels = ['λ = '+str(λ) for λ in λs]
#     if simulate:
#         γ = 1.0
#         ε = 0.0
#         αs = [0.12 + i*0.14 for i in range(4)]
#         max_steps = 1e4
#         runs = 2
#         errors = np.zeros((len(λs), len(αs), runs))
#         for run in range(runs):
#             for αi, α in tqdm(enumerate(αs)):
#                 for λi, λ in enumerate(λs):
#                     cartpole = CartPole(α, λ, γ, ε)
#                     steps = 0
#                     failures = 0
#                     while steps < 1e4:
#                         if steps%50==0: print("check"+str(steps) + 'failures = '+str(failures)+', predicted = '+str(failures*1e5/(steps+1)))
#                         steps += cartpole.sarsaλ(max_steps, replacing_traces=True)
#                         failures += 1
#                     errors[λi][αi][run] = failures
#         data = {l:(αs, list(np.mean(errors[li], axis=1))) for li, l in enumerate(labels)}
#         # data = {"xlab":"α", "ylab":"Failures per 100,000 steps", "title":"Cart & Pole", "ylims":[50, 300],
#         #        labels[0]:(λs, failures[1], failures[0], failures[2])}
#         export_dataset(data, "testData")
#     data = load_dataset("testData")
#     multipleCurvesPlot(data, "cart pole", 'α', 'failures per 100,000 timesteps',
#                     "test", w=6.0, h=4.5, labels=labels)#, ylims=[220, max_steps])


def figure_12_14(simulate=[False, False, False, False]):
    d1 = figure_12_14_MountainCar(simulate[0])
    d2 = figure_12_14_RandomWalk(simulate[1])
    d3 = figure_12_14_PuddleWorld(simulate[2])
    d4 = figure_12_14_CartAndPole(simulate[3])

    data = [d1, d2, d3, d4]
    plot_4_figs(data, title="", filename='plots/figure_12_14.png')




figure_12_10(simulate=False)
figure_12_11(simulate=True)
figure_12_14(simulate=[False, False, False, False])
cartpole_rec_episode()
