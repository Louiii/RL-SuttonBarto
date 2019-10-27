from OnPolicy_PredictionApproximation import *

def figure_9_1():
    agent = Agent(Walk(), VParams=[10])
    episodes = 100000
    agent.α = 0.00002
    distribution = np.zeros(agent.env.N + 2)
    for ep in tqdm(range(episodes)):
        agent.firstVisitMC_V_Episode(distribution)

    distribution /= np.sum(distribution)
    V = [agent.apxV.value(i) for i in agent.env.states]

    labels = ['Approximate MC value', 'True value']
    data = {labels[0]:(agent.env.states, V), 
            labels[1]:(agent.env.states, true_V[1: -1])}
    multipleCurvesPlot(data, "", "State", "Value", 'figure_9_1a', lgplc='upper left',
                        w=5.0, h=4.5, labels=[], ylims=None)

    labels = ['State distribution']
    data = {labels[0]:(agent.env.states, distribution[1: -1])}
    multipleCurvesPlot(data, "", "State", "Distribution", 'figure_9_1b', lgplc='upper left',
                        w=5.0, h=4.5, labels=[], ylims=None)


def figure_9_2_a():
    agent = Agent(Walk(), VParams=[10])
    episodes = int(1e5)
    agent.α = 2e-4
    agent.γ = 1
    agent.n = 1
    for ep in tqdm(range(episodes)):
        agent.semi_gradient_temporal_difference(rnd_policy=True)

    v = [agent.apxV.value(i) for i in agent.env.states]
    labels = ['Approximate TD value', 'True value']
    data = {labels[0]:(agent.env.states, v), 
            labels[1]:(agent.env.states, true_V[1: -1])}
    multipleCurvesPlot(data, "", "State", "Value", 'figure_9_2a', lgplc='upper left',
                        w=5.0, h=4.5, labels=[], ylims=None)

def figure_9_2_b(simulate=True):
    steps = np.power(2, np.arange(0, 10))
    labels = ['n = ' + str(steps[i]) for i in range(len(steps))]
    if simulate:
        αs = np.arange(0, 1.1, 0.1)
        episodes = 10
        runs = 100

        errors = np.zeros((len(steps), len(αs)))
        for run in tqdm(range(runs)):
            for step_ind, step in zip(range(len(steps)), steps):
                for α_ind, α in zip(range(len(αs)), αs):
                    agent = Agent(Walk(), VParams=[20])
                    agent.α = α
                    agent.γ = 1
                    agent.n = step
                    for ep in range(episodes):
                        agent.semi_gradient_temporal_difference(rnd_policy=True)
                        V = np.array([agent.apxV.value(i) for i in agent.env.states])
                        errors[step_ind, α_ind] += np.sqrt(np.sum(np.power(V - true_V[1: -1], 2)) / agent.env.N)
        errors /= episodes * runs
        data = {labels[i]:(list(αs), list(errors[i, :])) for i in range(len(steps))}
        export_dataset(data, 'figure_9_2b')

    data = load_dataset('figure_9_2b')
    multipleCurvesPlot(data, "", "α", "RMS Error", 'figure_9_2b', lgplc='lower left',
                        w=5.0, h=4.5, labels=labels, ylims=[0.25, 0.55])
    

def figure_9_5(true_V, simulate=True):
    orders = [5, 10, 20]
    αs = [0.0001, 0.00005]
    value_functions = ["polynomial basis", "fourier basis"]
    if simulate:
        runs = 1
        episodes = 5000
        errors = np.zeros((len(αs), len(orders), episodes))
        for run in range(runs):
            for i in range(len(orders)):
                for j in range(len(value_functions)):
                    agent = Agent(Walk(), vfn_type="Bases", VParams=[orders[i], value_functions[j]])
                    agent.α = αs[j]
                    for episode in tqdm(range(episodes)):
                        agent.firstVisitMC_V_Episode()
                        V = np.array([agent.apxV.value(state) for state in agent.env.states])
                        errors[j, i, episode] += np.sqrt(np.mean(np.power(true_V[1: -1] - V, 2)))
        errors /= runs

        # Store the data from the simulation
        export_dataset(list([[list(xij) for xij in xi] for xi in errors]), "fig9_5data")

    # Load the data from a simulation
    errors = np.array(load_dataset("fig9_5data"))
    data = {}
    labels = []
    for i in range(len(αs)):
        for j in range(len(orders)):
            l = '%s order = %d' % (value_functions[i], orders[j])
            labels.append(l)
            data[l] = (range(1, len(errors[i, j, :])+1), errors[i, j, :])
    multipleCurvesPlot(data, "Polynomial & Fourier bases for value function", 'Episodes', '$\sqrt{VE}$', "figure_9_5", w=6.0, h=4.5, labels=labels)

def figure_9_10(true_V, simulate=True):
    labels = ['tile coding (50 tilings)', 'state aggregation (one tiling)']
    if simulate:# takes about 3 hours
        runs = 30
        episodes = 5000

        num_tilings = 50
        tile_width = 200

        tiling_offset = 4

        env = Walk()
        value_functions = [("Tile", [num_tilings, tile_width, tiling_offset]), ("Aggregation", [env.N // tile_width])]
        errors = np.zeros((len(labels), episodes))
        for run in range(runs):
            for i in range(len(value_functions)):
                agent = Agent(Walk(), vfn_type=value_functions[i][0], VParams=value_functions[i][1])
                for episode in tqdm(range(episodes)):
                    agent.α = 1 / (episode + 1)
                    agent.firstVisitMC_V_Episode()
                    V = [agent.apxV.value(state) for state in agent.env.states]

                    errors[i][episode] += np.sqrt(np.mean(np.power(true_V[1: -1] - V, 2)))
        errors /= runs

        # Store the data from the simulation
        export_dataset(list([list(xi) for xi in errors]), "fig9_10data")

    # Load the data from a simulation
    errors = np.array(load_dataset("fig9_10data"))

    data = {labels[i] : (range(1, len(errors[i])+1), errors[i]) for i in range(len(labels))}
    multipleCurvesPlot(data, "Tile Coding vs State Aggregation", 'Episodes', '$\sqrt{VE}$', "figure_9_10", w=6.0, h=4.5, labels=labels)


# env = GridWorldCts()
#
# plotCts(env.costAll, env.w, env.h, show=True, title="Continuous policy", filename="plottingTest")
#

# env = Walk()
# export_dataset(list(env.compute_true_V()), "TrueValuesOfRandomWalkEnv")
true_V = np.array(load_dataset("TrueValuesOfRandomWalkEnv"))

figure_9_1()
figure_9_2_a()
figure_9_2_b()
figure_9_5()#simulate=False)
figure_9_10()#simulate=False)

