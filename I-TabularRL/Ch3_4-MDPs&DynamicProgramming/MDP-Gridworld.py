from modules import * #just numpy, grid environment and files in the folder gif_printer

class Agent():
    def __init__(self, env, γ=0.9):
        self.env = env
        self.V = np.zeros((env.w,env.h))
        self.πgreedy = {s: "" for s in env.states}
        self.γ = γ
        self.p = 1/len(env.actions)

    def updateπgreedy(self):
        for s in self.env.states:
            (i,j)=s
            if s==self.env.goal:
                pass
            elif i==self.env.w-1:# we are at the right side
                self.πgreedy[(i,j)]="up"
            elif j==self.env.h-1:# we are at the top
                self.πgreedy[(i,j)]="right"
            else:
                if self.V[i+1, j] > self.V[i, j+1]: # right > up
                    self.πgreedy[(i,j)]="right"
                else:
                    self.πgreedy[(i,j)]="up"

    def valueIterate(self, iterations_to_record=None, rec=True, updateπ=False):
        iteration = 0
        while True:# iterate until convergence
            iteration+=1
            v = np.zeros((self.env.w, self.env.h))
            for s in self.env.states:
                # values=[]
                for a in self.env.actions:
                    next_s, r = self.env.step(s, a)

                    # values.append( r + self.γ*self.V[next_s] )
                    v[s] += self.p * (r + self.γ*self.V[next_s])# Bellman equation
                # v[s] = max(values)

            if updateπ: self.policyImprovement()
            if rec:
                self.updateπgreedy()
                if iteration == iterations_to_record[0]:
                    record(iteration, "Equiprobable action policy with state values, iteration: ", v, self.πgreedy, self.env.w, self.env.h, self.env.costs)
                    _ = iterations_to_record.pop(0)

                if np.sum(np.abs(self.V - v)) < 1e-3:
                    record(iteration, "Equiprobable action policy with state values, iteration: ", v, self.πgreedy, self.env.w, self.env.h, self.env.costs)
                    break
            else:
                if np.sum(np.abs(self.V - v)) < 1e-3: break
            self.V = v



    def policyImprovement(self):
        ''' fn from policy iteration file
        used to compute the policy for figure 4.1
        '''
        policyStable = True
        for s in self.env.states:
            b = self.πgreedy[s]

            bellman = lambda sPrime_r: sPrime_r[1] + self.γ * self.V[sPrime_r[0]]
            action_reward = {a: bellman(self.env.step(s, a)) for a in self.env.actions}
            self.πgreedy[s] = max(action_reward, key=action_reward.get)# max action

            if b != self.πgreedy[s]: policyStable = False
        return policyStable

class Agent2():
    def __init__(self, env, γ=0.9):
        self.env = env
        self.V = np.zeros((env.w,env.h))
        self.π = {s: 0 for s in env.states}
        self.γ = γ
        self.p = poisson

    def valueIterate(self, iterations_to_record=None, rec=True, updateπ=False):
        iteration = 0
        while True:# iterate until convergence
            iteration+=1
            v = np.zeros((self.env.w, self.env.h))
            for s in self.env.states:
                # values=[]
                for a in self.env.actions:
                    # next_s, r = self.env.step(s, a)

                    p, r = self.env.p_next_state(s, a)

                    # values.append( r + self.γ*self.V[next_s] )
                    v[s] += sum([p[next_s] * (r[next_s] + self.γ*self.V[next_s]) for next_s in self.env.states])# Bellman equation
            self.policyImprovement()
            if np.sum(np.abs(self.V - v)) < 1e-3: break
            self.V = v

    def policyImprovement(self):
        ''' fn from policy iteration file
        used to compute the policy for figure 4.2
        '''
        policyStable = True
        for s in self.env.states:
            b = self.π[s]

            bellman = lambda sPrime_r: sPrime_r[1] + self.γ * self.V[sPrime_r[0]]
            action_reward = {a: bellman(self.env.step(s, a)) for a in self.env.actions}
            self.π[s] = max(action_reward, key=action_reward.get)# max action

            if b != self.π[s]: policyStable = False
        return policyStable

if __name__=="__main__":
    # ## figure 3.2
    env = TeleportGridWorld()
    agent = Agent(env)
    agent.valueIterate(rec=False)
    agent.env.plotgrid(agent.V, "figure_3_2")

    # ## figure 4.1
    env = SmallGridWorld()
    agent = Agent(env, γ=1)
    agent.valueIterate(rec=False)
    agent.policyImprovement()
    agent.env.plotgrid(agent.V, "figure_4_1", agent.πgreedy)

    ## figure 4.2
    # env = CarRental()
    # agent = Agent2(env)
    # agent.policyIteration(rec=False)
    # print(agent.π)

    # My GridWorld
    iterations_to_record = list( np.concatenate([np.arange(1,10,1),np.arange(10,30,2),np.arange(30,50,4),
        np.arange(50,100,10),np.arange(100,200,20),np.arange(200,500,40),np.arange(500,1000,50),np.arange(1000,2001,200)]) )

    env = GridWorld()
    agent = Agent(env)
    agent.valueIterate(iterations_to_record)

    makeGIF('../gif_printer/temp-plots', 'gifs/mdp', 0.25, 12)# framerate=0.25s, 12 repeats at the end
