from modules import *

class Model():
    def __init__(self, S, A):# give the action space and the state space
        self.step_apprx = {(s,a):[] for s in S for a in A}
        self.observed = set([])

    def update(self, s, a, s_dash, r):
        self.step_apprx[(s,a)] = [s_dash, r]
        self.observed.add((s, a))

    def sample(self):
        (s,a) = list(self.observed)[np.random.randint(len(self.observed))]
        [s_dash, r] = self.step_apprx[(s,a)]
        return s, a, s_dash, r

class TimeHeuristicModel():
    def __init__(self, S, A, κ=1e-4):# give the action space and the state space
        self.step_apprx = {(s,a):[] for s in S for a in A}
        self.time = 0
        self.κ = κ
        self.observed = set([])
        self.A = A

    def update(self, s, a, s_dash, r):
        self.time += 1
        if any([(s,ac) not in self.observed for ac in self.A]):
            for ac in self.A:
                if ac != a:
                    self.step_apprx[(s,ac)] = [s, 0, 1]
                    self.observed.add((s, ac))
        self.observed.add((s, a))
        self.step_apprx[(s,a)] = [s_dash, r, self.time]

    def sample(self):
        (s,a) = list(self.observed)[np.random.randint(len(self.observed))]
        [s_dash, r, time] = self.step_apprx[(s,a)]
        r += self.κ * np.sqrt(self.time - time)
        return s, a, s_dash, r

class PriorityModel():
    def __init__(self, S, A, θ=1e-4):
        self.step_apprx = {(s,a):[] for s in S for a in A}
        self.predecessors = {s:set([]) for s in S}
        self.observed = set([])
        self.PQueue = MinPriorityQueue()
        self.θ = θ

    def put(self, P, s, a):
        self.PQueue.add((s, a), -P)

    def empty(self):
        return self.PQueue.empty()

    def update(self, s, a, s_dash, r):
        self.step_apprx[(s,a)] = [s_dash, r]
        self.observed.add((s, a))
        self.predecessors[s_dash].add( (s, a) )

    def sample(self):
        (s, a), P = self.PQueue.get()
        s_dash, r = self.step_apprx[(s, a)]
        return -P, s, a, s_dash, r

class Agent():
    def __init__(self, env, model="", γ=0.95, α=0.1, ε=0.1, θ=1e-4, planning_steps=50, mxstp=float('inf'), κ=1e-4):
        self.env=env
        self.mtype=model
        if model=='dynamic':
            self.model = TimeHeuristicModel(env.states, env.actions, κ)#dynamic env
        elif model=='PSweep':
            self.model = PriorityModel(env.states, env.actions, θ=θ)
        else:
            self.model = Model(env.states, env.actions)
        self.Q = {(s,a):0 for s in env.states for a in env.actions}
        self.V = {s:0 for s in env.states}

        self.π = {s: np.random.choice(env.actions) for s in env.states}
        self.Π = {(s,a): 1/len(env.actions) for s in env.states for a in env.actions}

        self.γ = γ
        self.α = α
        self.ε = ε
        self.planning_steps = planning_steps # n-step planning

        self.max_steps = mxstp

    def sampleΠ(self, state):# choose an action over prob dist Π
        pmf = {a: self.Π[(state,a)] for a in self.env.actions}
        return np.random.choice(list(pmf.keys()), p=list(pmf.values()))

    def generateSoftPolicy(self):
        if max(self.Q.values()) != min(self.Q.values()):
            for s in self.env.states:
                x = {a: self.Q[(s,a)] for a in self.env.actions}
                self.π[s] = max(x, key=x.get)
            for s in self.env.states:
                for a in self.env.actions:
                    if a == self.π[s]:
                        self.Π[(s,a)] = 1 - self.ε + self.ε/len(self.env.actions)
                    else:
                        self.Π[(s,a)] = self.ε/len(self.env.actions)

    def tabularDynaQEpisode(self, start):
        steps = 0# number of steps in episode
        state = start
        while state not in self.env.goals and steps < self.max_steps:
            # if env_type == "maze":
            #     if state in self.env.goals or steps > self.max_steps: break

            steps += 1
            action = self.sampleΠ(state)
            next_state, reward = self.env.step(state, action)
            self.Q[(state, action)] += self.α*(reward + self.γ*max([self.Q[(next_state, a)] for a in self.env.actions]) - self.Q[(state, action)])

            # improve our model by adding the experience
            self.model.update(state, action, next_state, reward)

            # sample experience from the model: s = some state we've seen before, a = some action we've previously taken from s
            for t in range(self.planning_steps):
                s, a, s_dash, r = self.model.sample()# samples over all (s,a) equally, but not in time model!
                self.Q[(s,a)] += self.α*(r+self.γ*max([self.Q[(s_dash, a)] for a in self.env.actions])-self.Q[(s,a)])

            self.generateSoftPolicy()# update policy (epsilon greedy)

            # if env_type == "grid":
            #     if self.env.deterministicπEpisodeStillGoing(state, action)==False: break#state!=self.env.goal

            state = next_state
        return steps, state

    def runNEpisodes(self, N, env_type, iterations_to_record=[]):
        stepss = []
        for iteration in range(1,N+1):
            # print("iteration = "+str(iteration))
            # Run episode:
            start = self.env.start if env_type == "maze" else self.env.states[np.random.randint(len(self.env.states)-1)]
            steps, state = self.tabularDynaQEpisode(start)
            stepss.append(steps)
            # Record:
            if iteration in iterations_to_record:
                if env_type == "maze":self.mz_rcd(iteration,"Maze DynaQ policy and state values, episode: ")
                elif env_type == "grid":self.gd_rcd(iteration,"Maze DynaQ policy and state values, episode: ")
            # if iteration>20000: break# or sum([q[(s,a)]-self.Q[(s,a)] for s in self.env.states for a in self.env.actions]) < 1e1:
        return stepss

    def prioritisedSweepingEpisode(self, start):# for deterministic environment
        step = 0
        state = start
        while state not in self.env.goals and step < self.max_steps:
            step += 1
            action = self.sampleΠ(state)
            next_state, reward = self.env.step(state, action)

            self.model.update(state, action, next_state, reward)

            P = np.abs( reward + self.γ*max([self.Q[(next_state, a)] for a in self.env.actions]) - self.Q[(state, action)] )

            if P > self.model.θ: self.model.put( P, state, action )

            for i in range(self.planning_steps):
                if self.model.empty(): break
                # (s, a) = self.model.get()[1]
                #
                # (s_dash, r) = self.model.step_apprx[(s,a)]
                _, s, a, s_dash, r = self.model.sample()

                self.Q[(s,a)] += self.α*(r+self.γ*max([self.Q[(s_dash, ac)] for ac in self.env.actions])-self.Q[(s,a)])
                for pre_sa in list(self.model.predecessors[s]):

                    predicted_r = self.model.step_apprx[ pre_sa ][1]
                    P = np.abs( predicted_r + self.γ*max([self.Q[(s, ac)] for ac in self.env.actions]) - self.Q[pre_sa] )

                    if P > self.model.θ:
                        self.model.put( P, pre_sa[0], pre_sa[1] )

            self.generateSoftPolicy()
            # if self.env.deterministicπEpisodeStillGoing(state, action)==False: break#state!=self.env.goal
            state = next_state
        return step, state

    def mz_rcd(self,iter,tt):
        for s in self.env.states: self.π[s] = max([(a,self.Q[(s,a)]) for a in self.env.actions], key=lambda x:x[1])[0]
        for s in self.env.states: self.V[s] = sum([self.Q[(s,a)] for a in self.env.actions])
        maze_record(iter, tt, self.V, self.π, self.env.w, self.env.h, self.env)

    def gd_rcd(self,iter,tt):
        for s in self.env.states: self.π[s] = max([(a,self.Q[(s,a)]) for a in self.env.actions], key=lambda x:x[1])[0]
        for s in self.env.states: self.V[s] = sum([self.Q[(s,a)] for a in self.env.actions])
        record(iter, tt, self.V, self.π, self.env.w, self.env.h, self.env.costs)

    def runSteps(self, n):
        stepss = [0]
        acc = 0
        while acc <= n:
            # Run episode:
            start = self.env.start
            steps, _ = self.tabularDynaQEpisode(start) if self.mtype!='PSweep' else self.prioritisedSweepingEpisode(start)
            acc += steps
            stepss.append(acc)
        return stepss
