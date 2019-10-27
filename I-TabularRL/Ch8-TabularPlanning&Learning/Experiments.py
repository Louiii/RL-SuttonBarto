from Dyna import *

def figure_8_2(simulate=True):
    print("Running fig 8.2")
    if simulate:
        print("Simulating...")
        labels = planningStepsSimulation([0,5,50], 20, 50, "figure_8_2", "maze")
    labels = [str(step)+" planning steps" for step in [0,5,50]]
    data = load_dataset("figure_8_2")
    multipleCurvesPlot(data, "Average learning curves for DynaQ agents", "Episodes", "Steps per episode", "figure_8_2", labels=labels)

def figure_8_3():
    print("Running fig 8.3 ...")
    gifs()

def planningStepsSimulation(steps, repeats, n_eps, dataname, env):
    labels = [str(step)+" planning steps" for step in steps]
    data = {l:np.array([0.0 for _ in range(n_eps)]) for l in labels}
    for step in steps:
        for _ in tqdm(range(repeats)):
            envi = Maze() if env_type=="maze" else Environment()
            agent = Agent(envi)
            agent.max_steps = float('inf')
            agent.planning_steps = step
            # state = agent.env.start
            # # make them find the goal
            # while state not in agent.env.goals: steps, state = agent.tabularDynaQEpisode(agent.env.start)
            y = agent.runNEpisodes(n_eps, env, [])
            # print(y)
            data[str(step)+" planning steps"] += np.array(y)/repeats
    # each step has value: (xs,ys)
    for step in steps: data[str(step)+" planning steps"] = (list(range(2,n_eps+1)),list(data[str(step)+" planning steps"])[1:])
    export_dataset(data, dataname)
    return labels

def figure_8_4(simulate=True):
    print("Running fig 8.4")
    data_name = "figure_8_4"
    if simulate:
        print("Simulating...")
        dynaVsDynaPlusSimulation(data_name, True, False, mxstps=3000, av=40)
    data = load_dataset(data_name)
    multipleCurvesPlot(data, "Average performance curves for DynaQ agents on blocking task", "Time steps", "Cumulative reward", "figure_8_4", 'lower right', 9, 5)

def dynaVsDynaPlusSimulation(name, blkg, srtct, mxstps=3000, av=20, pstps=10, κ=1e-4):
    reward_all = np.zeros((2, mxstps))#[[0 for i in range(mxstps)],[0 for i in range(mxstps)]]
    for _ in tqdm(range(av)):
        dyna_agent = Agent(Maze(blocking=blkg, shortcut=srtct), model='', α=1.0, planning_steps=pstps, mxstp=3000, κ=κ)
        steps1 = dyna_agent.runSteps(mxstps)
        dynaPlus_agent = Agent(Maze(blocking=blkg, shortcut=srtct), model='dynamic', α=1.0, planning_steps=pstps, mxstp=3000, κ=κ)
        steps2 = dynaPlus_agent.runSteps(mxstps)
        st = [steps1, steps2]
        for j in range(2):
            curr = 0
            for i in range(1, len(st[j])):
                for idx in range(st[j][i-1],min(st[j][i],mxstps)):
                    reward_all[j][ idx ] = curr
                curr += 1
    labels = ["Dyna", "Dyna +"]
    data = {l:(list(range(1,mxstps+1)), list(reward_all[i])) for i, l in enumerate(labels)}
    export_dataset(data, name)

def figure_8_5(simulate=True):
    print("Running fig 8.5")
    data_name = "figure_8_5"
    if simulate:
        print("Simulating...")
        dynaVsDynaPlusSimulation(data_name, False, True, mxstps=6000, av=10, pstps=50, κ=1e-3)
    data = load_dataset(data_name)
    multipleCurvesPlot(data, "Average performance curves for DynaQ agents on shortcut task", "Time steps", "Cumulative reward", "figure_8_5", 'lower right', 9, 5)

def example_8_4(simulate=True):
    print("Running example 8.4")
    if simulate:
        print("Simulating...")
        mxstps, av = 3000, 10
        reward_all = [[0 for i in range(mxstps)],[0 for i in range(mxstps)]]
        for j in range(av):
            # print(j)
            prioritisedSweepingAgent = Agent(Maze(), model='PSweep', α=0.5, θ=1e-4, planning_steps=5, κ=1e-4)
            steps1 = prioritisedSweepingAgent.runSteps(mxstps)
            agent = Agent(Maze(), model='', α=0.5, θ=1e-4, planning_steps=5, κ=1e-4)
            steps2 = agent.runSteps(mxstps)
            st = [steps1, steps2]
            for j in range(2):
                curr = 0
                for i in range(1, len(st[j])):
                    for idx in range(st[j][i-1],min(st[j][i],mxstps)):
                        reward_all[j][ idx ] = curr
                    curr += 1
        data = {"prioritisedSweepingAgent":(list(range(1,mxstps+1)), reward_all[0]), "Dyna":(list(range(1,mxstps+1)), reward_all[1])}
    
    data_name="example_8_4"
    export_dataset(data, data_name)
    data = load_dataset(data_name)
    multipleCurvesPlot(data, "Average performance curves for DynaQ and prioritised sweeping agents on maze", "Time steps", "Cumulative reward", data_name, 'lower right', 9, 5)

def gifs(env_type="maze"):
    iterations_to_record = list( np.concatenate([np.arange(1,10,1),np.arange(10,100,10),np.arange(500,1000,100),np.arange(1000,20000,1000)]) )
    env = Maze() if env_type=="maze" else Environment()
    agent = Agent(env, model='')
    _ = agent.runNEpisodes(500, env_type, iterations_to_record)
    # agent.prioritisedSweeping(iterations_to_record)
    makeGIF('../gif_printer/temp-plots', 'gifs/figure_8_3', 0.5, 12)# framerate=0.25s, 12 repeats at the end
    print("Done.")

if __name__=="__main__":
    # env_type = ["maze", "grid"]

    figure_8_2(simulate=False)
    figure_8_3()
    figure_8_4(simulate=False)# blocking maze
    figure_8_5(simulate=False)# shortcut maze
    example_8_4(simulate=False)# Prioritised sweeping
    