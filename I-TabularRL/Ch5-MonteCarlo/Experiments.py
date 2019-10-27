# from OnPolicyMonteCarloAlgorithms import *
# from OffPolicyMonteCarloAlgorithms import *
from modules import *

def episode():
    ρ = 1
    trajectory = []
    while True:
        action = "left" if np.random.random() < 0.5 else "right"# b
        ρ *= 0.5
        if action == "right": return 0
        if np.random.random() < 0.1: return 1/ρ


def figure_5_4():
    repeats = 10
    x = 5
    epis = 10**x
    data = {i:None for i in range(repeats)}
    for run in tqdm(range(repeats)):
        tot=0
        rewards=[]
        for i in range(1, epis+1):
            tot += episode()
            rewards.append(tot/i)
        data[run] = (list(range(epis)), list(rewards))
    multipleCurvesPlot(data, "Infinite Variance", '(log) Episodes', 'MC estimate of $v_{\pi}$ ordinary\nimportance sampling over 10 runs', "figure_5_4", lgplc='upper right',
                       w=6.0, h=4.5, xlims=[1, epis], ylims=[0,4], labels=list(range(repeats)), xlog=([10**i for i in range(x+1)],[str(10**i) for i in range(x+1)]), leg=False)


if __name__=="__main__":
    figure_5_4()
