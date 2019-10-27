from modules import * #just numpy, grid environment and files in the folder gif_printer

class Agent():
    def __init__(self, env, γ=0.9):
        self.env = env
        self.V = np.zeros((env.w,env.h))
        self.π = {s:np.random.choice(env.actions) for s in env.states}
        self.γ = γ

    def policyEvaluation(self, θ=1e-2):
        Δ = θ + 1
        while Δ > θ:
            Δ = 0
            for s in self.env.states:
                v = self.V[s]
                # v     does r need to be a fn of π?
                # print("state = ",str(s),", π[s] = ",str(self.π[s]))
                s_prime, r = self.env.step(s, self.π[s])# π is deterministic
                # print("s_prime = ",str(s_prime),", r = ",str(r))
                self.V[s] = r + self.γ * self.V[s_prime]
                #sum([ p*(r + self.γ*self.V[s_prime]) for a in actions with s_prime, r = self.env.step(s, a)])
                Δ = max( abs(v - self.V[s]), Δ )
            # print(Δ)

    def policyImprovement(self):
        policyStable = True
        for s in self.env.states:
            b = self.π[s]

            bellman = lambda sPrime_r: sPrime_r[1] + self.γ * self.V[sPrime_r[0]]
            action_reward = {a: bellman(self.env.step(s, a)) for a in self.env.actions}
            self.π[s] = max(action_reward, key=action_reward.get)# max action

            if b != self.π[s]: policyStable = False
        return policyStable

    def policyIteration(self, rec=True):
        policyStable = False
        iteration = 0
        while policyStable == False:
            self.policyEvaluation()
            policyStable = self.policyImprovement()
            # print(self.V)

            iteration += 1
            if rec: record(iteration, "Policy iteration with state values, iteration: ", self.V, self.π, self.env.w, self.env.h, self.env.costs)

if __name__=="__main__":
    ## figure 3.5
    env = TeleportGridWorld()
    agent = Agent(env)
    agent.policyIteration(rec=False)
    agent.env.plotgrid(agent.V, "figure_3_5", agent.π)

    # figure 4.1
    env = SmallGridWorld()
    agent = Agent(env)
    agent.policyIteration(rec=False)
    agent.env.plotgrid(agent.V, "figure_4_1", agent.π)

    # My GridWorld
    env = GridWorld()
    agent = Agent(env)
    agent.policyIteration()

    makeGIF('../gif_printer/temp-plots', 'gifs/PolicyIteration', 0.5, 12)# framerate=0.25s, 12 repeats at the end
