from PolicyGradientMethods import *

def figure_13_1(simulate=True):
    αs = [2**-12, 2**-13, 2**-14]
    labels = ["α = $2^{-12}$", "α = $2^{-13}$", "α = $2^{-14}$"]
    if simulate:
        num_eps = 1000
        repeats = 100
        G_episode = np.zeros((len(αs), num_eps))
        for run in tqdm(range(repeats)):
            for αi, α in enumerate(αs):
                agent = Agent(SCGridWorld(), α, 1, 2)
                for episode in range(num_eps):
                    G0 = agent.REINFORCE()
                    G_episode[αi, episode] += G0/repeats
        data = {l:(list(range(1000)), list(G_episode[li, :])) for li, l in enumerate(labels)}
        export_dataset(data, "figure_13_1")
    data = load_dataset("figure_13_1")

    data.update({"$v_*(s_0)$":([0, 1000], [-11.5858, -11.5858])})
    multipleCurvesPlot(data, "MC REINFORCE", "Episode", "$G_0$\nTotal reward on episode\naveraged over 100 runs", "figure_13_1", lgplc='lower right',
                            w=10.0, h=4.5, labels=labels+["$v_*(s_0)$"], ylims=None)

def figure_13_2(simulate=True):
    αθ, αw = 2**-9, 2**-6
    labels = ["REINFORCE with baseline $α^θ$ = $2^{-9}$, $α^w$ = $2^{-6}$"]
    if simulate:
        num_eps = 1000
        repeats = 100
        G_episode = np.zeros(num_eps)
        for run in tqdm(range(repeats)):
            agent = Agent(SCGridWorld(), αθ, 1, 2, nαw=(αw, 1))
            for episode in range(num_eps):
                G0 = agent.baselineREINFORCE()
                G_episode[episode] += G0/repeats
        data = {labels[0]:(list(range(1000)), list(G_episode))}
        export_dataset(data, "figure_13_2")

    data = load_dataset("figure_13_2")
    d = load_dataset("figure_13_1")
    data.update( {"REINFORCE α = $2^{-13}$":d["α = $2^{-13}$"]} )
    data.update({"$v_*(s_0)$":([0, 1000], [-11.5858, -11.5858])})
    multipleCurvesPlot(data, "Baseline MC REINFORCE", "Episode", "$G_0$\nTotal reward on episode\naveraged over 100 runs", "figure_13_2", lgplc='lower right',
                            w=10.0, h=4.5, labels=labels+["$v_*(s_0)$", "REINFORCE α = $2^{-13}$"], ylims=None)




figure_13_1(simulate=False)
figure_13_2()
