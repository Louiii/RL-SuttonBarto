from EligibilityTraces import *


λs = [0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
n_episodes = 10
repeats = 50

def run_simulation(λαs, fname, fn):
    errors = np.zeros((len(λs), len(λαs[λs[0]])))
    for λi, λ in tqdm(enumerate(λs)):
        for αi, α in enumerate(λαs[λ]):
            for i in range(repeats):
                agent = predAgent(Walk(), n_weights=21, λ=λ, α=α)
                for _ in range(n_episodes):
                    episode = getattr(agent, fn)
                    episode()
                    # compute error:
                    difference = np.array([agent.V.value(s) for s in agent.env.states]) - agent.env.true_V[1:-1]
                    errors[λi, αi] += np.sqrt(np.mean(np.power(difference, 2)))/(repeats*n_episodes)
    # average_rms = np.mean(errors, 2)/n_episodes# 2 is the axis for repeats
    data = {"λ = "+str(λ):(list(λαs[λ]), list(errors[λi, :])) for λi, λ in enumerate(λs)}
    export_dataset(data, fname)

def figure_12_3(simulate=True):
    if simulate:
        λαs = {λ: np.linspace(0, 1, 21) if λ <= 0.95 else np.linspace(0, (1.006-λ)*17-0.01, 21) for λ in λs}
        run_simulation(λαs, "figure_12_3", "OfflineSemiGradientλReturn_episode")
    data = load_dataset("figure_12_3")
    multipleCurvesPlot(data, "Offline TD(λ)", "α", "RMS error at the end of the episode over the first 10 episodes",
                        "figure_12_3", w=6.0, h=4.5, labels=["λ = "+str(λ) for λ in λs], ylims=[0.22, 0.55])


def figure_12_6(simulate=True):
    if simulate:
        λαs = {λ: np.linspace(0, 1, 21) if λ < 0.8 else np.linspace(0, 3*np.sqrt(1.05-λ)-0.62, 21) for λ in λs}
        run_simulation(λαs, "figure_12_6", "semiGradientTDλ_episode")
    data = load_dataset("figure_12_6")
    multipleCurvesPlot(data, "TD(λ)", "α", "RMS error at the end of the episode over the first 10 episodes",
                        "figure_12_6", w=6.0, h=4.5, labels=["λ = "+str(λ) for λ in λs], ylims=[0.25, 0.55])

def figure_12_8(simulate=True):
    if simulate:
        λαs = {λ: np.linspace(0, 1, 21) for λ in λs}
        λαs[0.975], λαs[0.99], λαs[1] = np.linspace(0, 0.85, 21), np.linspace(0, 0.35, 21), np.linspace(0, 0.1, 21)
        run_simulation(λαs, fname="figure_12_8", fn="trueOnlineTD")
    data = load_dataset("figure_12_8")
    multipleCurvesPlot(data, "True Online TD(λ)", "α", "RMS error at the end of the episode over the first 10 episodes",
                        "figure_12_8", w=6.0, h=4.5, labels=["λ = "+str(λ) for λ in λs], ylims=[0.25, 0.55])

def monteDutchFig(simulate=True):
    if simulate:
        λαs = {λ: np.linspace(0, 1, 21) for λ in λs}
        λαs[0.975], λαs[0.99], λαs[1] = np.linspace(0, 0.85, 21), np.linspace(0, 0.35, 21), np.linspace(0, 0.1, 21)
        run_simulation(λαs, fname="monteDutch", fn="monteCarloDutchTraces")
    data = load_dataset("monteDutch")
    multipleCurvesPlot(data, "Monte Carlo Dutch Traces", "α", "RMS error at the end of the episode over the first 10 episodes",
                        "monteDutch", w=6.0, h=4.5, labels=["λ = "+str(λ) for λ in λs], ylims=[0.25, 0.55])


figure_12_3(simulate=False)#simulate takes about 4 hours (with 50 repeats)
figure_12_6()
figure_12_8()
# monteDutchFig()
