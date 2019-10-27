from Bootstrapping import *

def fig7_2(simulate=True):
    ns = np.power(2, np.arange(0, 10))
    if simulate:
        repeats = 100
        episodes = 10
        αs = np.concatenate([np.array([0, 0.025, 0.05, 0.075]), np.arange(0.1, 1.1, 0.1)] )#np.arange(0, 1.1, 0.1)

        # track the errors for each (step, alpha) combination
        errors = np.zeros((len(ns), len(αs)))
        for _ in tqdm(range(repeats)):
            for αi, α in zip(range(len(αs)), αs):
                for ni, n in zip(range(len(ns)), ns):
                    env = Walk()
                    # print(env.true_V)
                    agent = Agent(env, γ=1, α=α, ε=1, n=n, V={s:0 for s in range(env.N_STATES+2)})
                    for episode in range(episodes):
                        agent.nStepTD_VPredictionEpisode(start=env.start, rnd_policy=True)
                        V = np.array([v for s,v in agent.V.items()])
                        # print(V)
                        errors[ni, αi] += np.sqrt(np.sum(np.power(V - env.true_V, 2)) / len(env.states))
        # take average
        errors /= episodes * repeats
        data = {'n = '+str(n):(list(αs), list(errors[ni,:])) for ni, n in zip(range(len(ns)), ns)}
        export_dataset(data, "ex7_2data")
    data = load_dataset("ex7_2data")
    multipleCurvesPlot(data, "n-Step TD errors for random walk task", "α",
                    "Average $\sqrt{\overline{SE}}$ over 19 states and first 10 episodes", "Fig7_2",
                    w=5.0, h=4.5, ylims=[0.25, 0.55], labels=['n = '+str(n) for n in ns])

if __name__=="__main__":
    fig7_2(simulate=False)

    iterations_to_record = list( np.concatenate([np.arange(1,3,1),np.arange(10,100,10),np.arange(500,1000,100),np.arange(1000,20000,1000)]) )
    
    env = GridWorld()
    agent = Agent(env)
    # agent.firstVisitMC_Q(iterations_to_record)
    agent.nStepSARSA_Control(iterations_to_record)
    
    makeGIF('../gif_printer/temp-plots', 'gifs/SARSA_Bootstrapping', 0.5, 12)# framerate=0.25s, 12 repeats at the end
