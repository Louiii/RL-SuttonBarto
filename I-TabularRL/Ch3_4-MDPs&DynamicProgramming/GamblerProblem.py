from modules import multipleCurvesPlot
import numpy as np

class GamblersProblem:
    def __init__(self):
        self.states = [i for i in range(1, 100)]
        self.terminal_state = [0, 100]



class Agent():
    def __init__(self, env, pH, γ=0.9):
        self.env = env
        self.V = np.zeros(101)
        self.V[100] = 1
        # self.π = {s:np.random.choice(env.actions) for s in env.states}
        self.γ = γ
        self.pH = pH

    def compute_expectation(self, s):
        A = np.arange(min(s, 100-s)+1)
        R = np.zeros(len(A))
        # R[A+s==100] = self.pH
        VH = self.V[s+A]*self.pH*self.γ
        VT = self.V[s-A]*(1-self.pH)*self.γ
        return VH+VT#+R

    def valueIteration(self, θ=1e-10):
        Δ = θ + 1
        i = 0
        sweep = {i:None for i in [1,2,3,10]}
        while Δ > θ:
            i += 1
            Δ = 0
            for s in self.env.states:
                v = self.V[s].copy()

                self.V[s] = np.max(self.compute_expectation(s))
                
                Δ = max( abs(v - self.V[s]), Δ )
            if i in sweep: sweep[i] = (list(range(len(self.V))), list(self.V.copy()))
        return sweep

    def compute_greedy(self):
        return [np.argmax(np.round(self.compute_expectation(s)[1:], 4)) for s in self.env.states]

if __name__=="__main__":
    env = GamblersProblem()
    agent = Agent(env, 0.4, γ=1.0)
    sweep = agent.valueIteration()
    labels = ["sweep "+str(k) for k in sweep]
    data = {"sweep "+str(k):sweep[k] for k in sweep}
    multipleCurvesPlot(data, "Gamblers Problem", "State", "Value estimates", "figure_4_3a", lgplc='lower right', w=6.0, xlims=[1,99], labels=labels)
    
    data = {"policy":(agent.env.states, agent.compute_greedy())}
    multipleCurvesPlot(data, "Policy", "State", "Final policy (stake)", "figure_4_3b", w=6.0)

