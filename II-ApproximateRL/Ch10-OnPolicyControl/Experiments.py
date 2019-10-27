from tqdm import tqdm
from math import floor
from OnPolicyControlApproximation import *

def figure_10_1():
    episodes = 9000
    plot_episodes = [1, 100, 1000, episodes]
    car = MountainCar()
    agent = Agent(car, QFnArgs=[0.3, 8])# α, num_of_tilings
    for episode in tqdm(range(episodes+1)):
        _=agent.semiGradient_nStep_sarsa_episode()
        if episode in plot_episodes:
            grid_size = 40
            positions = np.linspace(agent.env.pos_bound[0], agent.env.pos_bound[1], grid_size)
            velocities = np.linspace(agent.env.vel_bound[0], agent.env.vel_bound[1], grid_size)
            axis_x = []
            axis_y = []
            axis_z = []
            for position in positions:
                for velocity in velocities:
                    axis_x.append(position)
                    axis_y.append(velocity)
                    state = (position, velocity)
                    axis_z.append(agent.apxQ.cost_to_go(state))

            surface_plot(axis_x, axis_y, axis_z, 'figure_10_1_ep'+str(episode), 
                'position', 'velocity', 'cost-to-go', ('Episode %d' % episode))

    surfaceRotation(axis_x, axis_y, axis_z, 'position', 'velocity', 'cost-to-go', ('Episode %d' % episodes))
    makeGIF('../gif_printer/temp-plots', 'gifs/figure_10_1_episode'+str(episodes), 0.1, 0)

    agent.env.position_list = []
    agent.semiGradient_nStep_sarsa_episode()
    agent.env.render(file_path='./gifs/mountain_car.gif', mode='gif')

def figure_10_2(simulate=True):
    αs = [0.1, 0.2, 0.5]
    num_of_tilings = 8
    if simulate:
        runs = 10
        episodes = 500
        steps = np.zeros((len(αs), episodes))
        for run in range(runs):
            # value_functions = [ValueFunction(α, num_of_tilings) for α in αs]
            for index, α in zip(range(len(αs)), αs):
                car = MountainCar()
                agent = Agent(car, QFnArgs=[α, num_of_tilings])
                for episode in tqdm(range(episodes)):
                    step = agent.semiGradient_nStep_sarsa_episode()
                    steps[index, episode] += step

        steps /= runs

        # Store the data from the simulation
        export_dataset(list([list(xi) for xi in steps]), "fig10_2data")

    # Load the data from a simulation
    steps = np.array(load_dataset("fig10_2data"))

    labels = ['α = '+str(α)+'/'+str(num_of_tilings) for α in αs]

    data = {labels[i] : (range(1, len(steps[i])+1), steps[i]) for i in range(len(labels))}
    multipleCurvesPlot(data, "", 'Episode', 'Steps per episode', "figure_10_2", w=6.0, h=4.5, labels=labels, log=True)

def figure_10_3(simulate=True):
    num_of_tilings = 8
    αs = [0.5, 0.3]
    n_steps = [1, 8]
    ε = 0
    if simulate:
        runs = 10
        episodes = 500

        steps = np.zeros((len(αs), episodes))
        for run in range(runs):
            for index, α in zip(range(len(αs)), αs):
                car = MountainCar()
                agent = Agent(car, ε=ε, n=n_steps[index], QFnArgs=[α, num_of_tilings])
                for episode in tqdm(range(episodes)):
                    step = agent.semiGradient_nStep_sarsa_episode()
                    steps[index, episode] += step

        steps /= runs

        # Store the data from the simulation
        export_dataset(list([list(xi) for xi in steps]), "fig10_3data")

    # Load the data from a simulation
    steps = np.array(load_dataset("fig10_3data"))

    labels = ['n = ' +str(n) for n in n_steps]

    data = {labels[i] : (range(1, len(steps[i])+1), steps[i]) for i in range(len(labels))}
    multipleCurvesPlot(data, "ε = "+str(ε), 'Episode', 'Steps per episode', "figure_10_3", w=6.0, h=4.5, labels=labels, log=True)

def figure_10_4(simulate=True):
    αs = [0.15*i for i in range(1, 12)]
    n_steps = [2**i for i in range(5)]
    labels = ['n = '+str(n) for n in n_steps]
    max_steps = 400
    if simulate:
        episodes = 50
        runs = 2# 100 takes about 2 hours
        steps = np.zeros((len(n_steps), len(αs)))
        for run in tqdm(range(runs)):
            for n_index, n in enumerate(n_steps):
                for α_index, α in enumerate(αs):
                    if (n==4 and α >= 1.5) or (n == 8 and α > 1) or (n == 16 and α >= 0.75):
                        steps[n_index, α_index] += max_steps * episodes
                    else:
                        # print('\nα = '+str(α)+', n = '+str(n))
                        car = MountainCar()
                        agent = Agent(car, n=n, QFnArgs=[α])
                        for episode in range(episodes):
                            step = agent.semiGradient_nStep_sarsa_episode()
                            steps[n_index, α_index] += step
        steps /= runs * episodes

        # Store the data from the simulation
        data = {labels[i] : (αs, list(steps[i])) for i in range(len(labels))}
        export_dataset(data, "fig10_4data")

    # Load the data from a simulation
    data = load_dataset("fig10_4data")
    multipleCurvesPlot(data, "", 'α * number of tilings(8)', 'Steps per episode',
                    "figure_10_4", w=6.0, h=4.5, labels=labels, ylims=[220, 300])

def figure_10_5(simulate=True):
    env = ServersEnvironment()
    labels = ["Priority "+str(env.rewards[p]) for p in env.priorities]

    if simulate:
        max_steps = int(2e6)
        # use tile coding with 8 tilings
        num_of_tilings = 8
        agent = Agent(env, servers_example=True, QFnArgs=[0.01, num_of_tilings])
        print("Running steps..")
        agent.differential_SemiGradient_sarsa_control_episode(max_steps)
        print("Done")
        values = np.zeros((len(env.priorities), env.n_servers + 1))
        for priority in env.priorities:
            for agent.env.available_servers in range(env.n_servers + 1):
                values[priority, agent.env.available_servers] = agent.apxQ.state_value((priority, agent.env.available_servers))

        data = {labels[i]:(list(range(env.n_servers + 1)), list(values[i, :])) for i in range(len(labels))}

        # save policy..
        policy = np.zeros((len(env.priorities), env.n_servers + 1))
        for priority in env.priorities:
            for agent.env.available_servers in range(1, env.n_servers + 1):
                values = [agent.apxQ.value((priority, agent.env.available_servers), action) for action in env.actions]
                if agent.env.available_servers == 0:
                    policy[priority, agent.env.available_servers] = 0# reject
                else:
                    policy[priority, agent.env.available_servers] = np.argmax(values)
        policy_data = {'policy':[list(p) for p in policy]}

        data.update(policy_data)
        export_dataset(data, 'figure_10_5')
    data = load_dataset('figure_10_5')
    multipleCurvesPlot(data, "", 'Number of free servers', 'Differential value of best action', "figure_10_5a", 
                        w=5.0, h=4.5, labels=labels)

    fig, ax = plt.subplots()
    policy = data['policy']
    fig = sns.heatmap(policy, cmap="YlGnBu", ax=ax, xticklabels=range(env.n_servers + 1), yticklabels=env.priorities)
    fig.set_title('Policy (0 Reject, 1 Accept)')
    fig.set_xlabel('Number of free servers')
    fig.set_ylabel('Priority')

    plt.savefig('plots/figure_10_5b.png', dpi=256)
    plt.close()

if __name__ == '__main__':
    figure_10_1()# and mountain car gif!
    figure_10_2()#(simulate=False)
    figure_10_3()#(simulate=False)
    figure_10_4()#(simulate=False)
    figure_10_5()#(simulate=False)
