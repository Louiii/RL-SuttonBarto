import matplotlib.pyplot as plt
from OffPolicyDivergenceExamples import *

def figure_11_2a():
    labels = ['θ$_{'+str(i)+'}$' for i in range(1, 9)]

    env = BairdsCounterExample()
    agent = Agent(env)
    steps = 1000
    θs = np.zeros((len(agent.V.θ), steps))
    state = np.random.choice(env.states)
    for step in range(steps):
        state, _ = agent.semiGradientOffPolicyTD_iteration(state)
        θs[:, step] = agent.V.θ

    
    data = { labels[i]: (list(range(1, 1001)), list(θs[i, :])) for i in range(len(agent.V.θ)) }
    multipleCurvesPlot(data, "Semi-Gradient Off-Policy TD", 'Step', 'θ', "figure_11_2a", lgplc='upper left', w=6.0, h=4.5, labels=labels, ylims=None)

def figure_11_2b():
    labels = ['θ$_{1}$-θ$_{6}$', 'θ$_{7}$', 'θ$_{8}$']
    # Initialize the theta
    env = BairdsCounterExample()
    agent = Agent(env)
    sweeps = 1000
    θs = np.zeros((len(agent.V.θ), sweeps))
    for sweep in range(sweeps):
        agent.semiGradientDP_sweep()
        θs[:, sweep] = agent.V.θ

    data = { labels[i]: (list(range(1, 1001)), list(θs[idx, :])) for i, idx in enumerate([5, 6, 7]) }
    multipleCurvesPlot(data, "Semi-Gradient DP", 'Sweeps', 'θ', "figure_11_2b", lgplc='upper left', w=6.0, h=4.5, labels=labels, ylims=None)


def figure_11_6a():
    labels = ['θ$_{'+str(i)+'}$' for i in range(1, 9)] + ['$\sqrt{\overline{VE}}$', '$\sqrt{\overline{PBE}}$']

    env = BairdsCounterExample()
    agent = Agent(env, α=0.005, β=0.05)
    steps = 1000
    θs = np.zeros((len(agent.V.θ), steps))
    RMSVE, RMSPBE = np.zeros(steps), np.zeros(steps)
    state = np.random.choice(env.states)

    v_t = np.zeros(len(agent.V.θ))

    for step in range(steps):
        state, reward = agent.GTD0(state, v_t)
        θs[:, step] = agent.V.θ
        RMSVE[step] = agent.compute_RMSVE()
        RMSPBE[step] = agent.compute_RMSPBE()

    data = { labels[i]: (list(range(1, 1001)), list(θs[i, :])) for i in range(8) }
    data.update({'$\sqrt{\overline{VE}}$': (list(range(1, 1001)), list(RMSVE)), 
                 '$\sqrt{\overline{PBE}}$':(list(range(1, 1001)), list(RMSPBE))})
    multipleCurvesPlot(data, "TDC", 'Steps', 'θ, Error metrics', "figure_11_6a", lgplc='upper right', w=6.0, h=4.5, labels=labels, ylims=None)

def figure_11_6b():
    labels = ['θ$_{'+str(i)+'}$' for i in range(1, 9)] + ['$\sqrt{\overline{VE}}$', '$\sqrt{\overline{PBE}}$']

    env = BairdsCounterExample()
    agent = Agent(env, α=0.005, β=0.05)

    sweeps = 1000
    θs = np.zeros((len(agent.V.θ), sweeps))
    v_t = np.zeros(len(agent.V.θ))
    RMSVE, RMSPBE = np.zeros(sweeps), np.zeros(sweeps)
    for sweep in range(sweeps):
        agent.expected_GTD0(v_t)
        θs[:, sweep] = agent.V.θ
        RMSVE[sweep] = agent.compute_RMSVE()
        RMSPBE[sweep] = agent.compute_RMSPBE()

    data = { labels[i]: (list(range(1, 1001)), list(θs[i, :])) for i in range(8) }
    data.update({'$\sqrt{\overline{VE}}$': (list(range(1, 1001)), list(RMSVE)), 
                 '$\sqrt{\overline{PBE}}$':(list(range(1, 1001)), list(RMSPBE))})
    multipleCurvesPlot(data, "Expected TDC", 'Sweeps', 'θ, Error metrics', "figure_11_6b", lgplc='upper right', w=6.0, h=4.5, labels=labels, ylims=None)

def figure_11_7():
    labels = ['θ$_{'+str(i)+'}$' for i in range(1, 9)] + ['$\sqrt{\overline{VE}}$']

    env = BairdsCounterExample()
    agent = Agent(env, α=0.03)

    sweeps = 1000
    θs = np.zeros((len(agent.V.θ), sweeps))
    RMSVE = np.zeros(sweeps)
    emphasis = 0.0
    for sweep in range(sweeps):
        emphasis = agent.expected_emphatic_TD(emphasis)
        θs[:, sweep] = agent.V.θ
        RMSVE[sweep] = agent.compute_RMSVE()

    data = { labels[i]: (list(range(1, 1001)), list(θs[i, :])) for i in range(8) }
    data.update({'$\sqrt{\overline{VE}}$': (list(range(1, 1001)), list(RMSVE))})
    multipleCurvesPlot(data, "Expected TDC", 'Sweeps', 'θ, Error metrics', "figure_11_7", lgplc='upper right', w=6.0, h=4.5, labels=labels, ylims=None)

if __name__=="__main__":
    figure_11_2a()
    figure_11_2b()
    figure_11_6a()
    figure_11_6b()
    figure_11_7()
