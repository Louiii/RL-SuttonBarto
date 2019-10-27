from modules import *
from FunctionApproximators import *


class SAR():
    def __init__(self, s, a, r):
        self.s = s
        self.a = a
        self.r = r

class AggregationValueFunction:
    def __init__(self, env, num_of_groups):
        self.env = env
        self.num_of_groups = num_of_groups
        self.group_size = env.N // num_of_groups
        self.θ = np.zeros(num_of_groups)

    def value(self, state):
        if state in self.env.goals: return 0
        i = (state - 1) // self.group_size
        return self.θ[i]

    def update(self, delta, state):
        i = (state - 1) // self.group_size
        self.θ[i] += delta

class BasesValueFunction:
    def __init__(self, env, order, type):
        self.env = env
        self.order = order
        self.weights = np.zeros(order + 1)
        self.bases = []
        if type == "polynomial basis":
            for i in range(order + 1):
                self.bases.append(lambda s, i=i: pow(s, i))
        if type == "fourier basis":
            for i in range(order + 1): 
                self.bases.append(lambda s, i=i: np.cos(i * np.pi * s))

    def value(self, state):
        state /= float(self.env.N)
        feature = np.array([func(state) for func in self.bases])
        return np.dot(self.weights, feature)

    def update(self, delta, state):
        state /= float(self.env.N)
        derivative_value = np.asarray([func(state) for func in self.bases])
        self.weights += delta * derivative_value

class TilingsValueFunction:
    def __init__(self, env, numOfTilings, tileWidth, tilingOffset):
        self.numOfTilings = numOfTilings
        self.tileWidth = tileWidth
        self.tilingOffset = tilingOffset
        self.tilingSize = env.N // tileWidth + 1
        self.θ = np.zeros((self.numOfTilings, self.tilingSize))
        self.tilings = np.arange(-tileWidth + 1, 0, tilingOffset)

    def value(self, state):
        stateValue = 0.0
        for tilingIndex in range(0, len(self.tilings)):
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            stateValue += self.θ[tilingIndex, tileIndex]
        return stateValue

    def update(self, delta, state):
        delta /= self.numOfTilings
        for tilingIndex in range(0, len(self.tilings)):
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            self.θ[tilingIndex, tileIndex] += delta

class RadialBasisFunction:
    def __init__(self, μs, σs):
        self.μs = μs
        self.σs = σs
        self.θ = np.zeros((self.numOfTilings, self.tilingSize))

    def value(self, state):
        return sum([ self.θ[i] * np.exp(-((state-self.μs[i])**2)/(2*self.σs[i]*2)) for i in range(len(self.θ))])

    def update(self, delta, state):
        x = np.array([np.exp(-((state-self.μs[i])**2)/(2*self.σs[i]**2)) for i in range(len(self.θ))])
        x /= sum(x)
        for i in range(len(self.θ)):
            self.θ[i] += delta * x[i]

class Agent():
    def __init__(self, env, γ=0.9, α=0.005, n=5, vfn_type="Aggregation", VParams=[]):
        self.env = env
        self.Q = {}

        if vfn_type=="Aggregation":
            self.apxV = AggregationValueFunction(env, *VParams) #LinearApprox()
        elif vfn_type=="Bases":
            self.apxV = BasesValueFunction(env, *VParams)
        elif vfn_type=="Tile":
            self.apxV = TilingsValueFunction(env, *VParams)
        self.α = α
        self.γ = γ
        self.n = n
        self.S = {i: None for i in range(n+1)}
        self.R = {i: 0 for i in range(n+1)}

    def SGD_update(self, U, s):
        self.apxV.ω += self.α * (U - self.apxV.V(s)) * self.apxV.gradient(s)

    def generateEpisode(self, start=None):
        """ Generate an episode following π:
        episode = [(S0, A0, R1), (S1, A1, R2),...,(S_T-1 , A_T-1, R_T)
        """
        state = self.env.states[np.random.choice(range(len(self.env.states)))] if start==None else start
        episode = []
        while state not in self.env.goals: # not self.env.goalState(state):
            action = random.choice(self.env.actions)#self.π[state]
            next_state, reward = self.env.step(state, action)
            episode.append( SAR(state, action, reward) )
            state = next_state
        return episode

    def firstVisitMC_V_Episode(self, distribution=None):
        # Generate an episode:
        episode = self.generateEpisode(self.env.start)

        # Update V
        G = 0
        # for (s,a,r) in reversed(episode):
        for i in reversed(range(len(episode))):
            s,r = episode[i].s, episode[i].r
            G = self.γ*G + r

            # self.SGD_update(G, s)
            delta = self.α * (episode[-1].r - self.apxV.value(s))
            self.apxV.update(delta, s)
            if distribution is not None:
                distribution[s] += 1
            # if s not in [sar.s for sar in episode[:i]]:# if this is the first occurence
            #     self.returns[s].append(G)
            #     self.V[s] = np.mean(self.returns[s])

    def semi_gradient_temporal_difference(self, start=None, rnd_policy=False):
        # initial starting state
        self.S[0] = start if start is not None else self.env.states[np.random.randint(len(self.env.states)-1)]

        t, τ = 0, 0 # track the time (and update time)
        T = np.inf # the length of this episode
        while τ != T - 1:
            if t < T:
                state = self.S[t%(self.n+1)]
                action = self.sampleΠ(state) if rnd_policy==False else np.random.choice(self.env.actions)
                next_state, reward = self.env.step(state, action)

                # Store in buffer...
                self.S[(t+1)%(self.n+1)] = next_state
                self.R[(t+1)%(self.n+1)] = reward

                if next_state in self.env.goals:
                    T = t+1

            τ = t - self.n + 1 # get the time of the state to update
            if τ >= 0:
                # calculate corresponding rewards, G is our returns
                G = sum([self.R[i%(self.n+1)] * self.γ**(i-τ-1) for i in range(τ+1, min(τ+self.n, T)+1)])
                if τ + self.n < T:
                    G += self.apxV.value(self.S[(τ+self.n)%(self.n+1)]) * self.γ**self.n

                state_to_update = self.S[τ%(self.n+1)]
                # update the value function
                if not state_to_update in self.env.goals:# goals have V = 0
                    delta = self.α * (G - self.apxV.value(state_to_update))
                    self.apxV.update(delta, state_to_update)
            t += 1 # go to next time step
            state = next_state

    def firstVisitMC_V(self,iterations_to_record):
        iteration = 0
        while True:
            # Generate an episode:
            episode = self.generateEpisode()

            # print(episode)
            print(iteration)
            v = np.copy(self.V)

            # Update V
            G = 0
            # for (s,a,r) in reversed(episode):
            for i in reversed(range(len(episode))):
                s,r = episode[i].s, episode[i].r
                G = self.γ*G + r

                self.SGD_update(G, s)
                # if s not in [sar.s for sar in episode[:i]]:# if this is the first occurence
                #     self.returns[s].append(G)
                #     self.V[s] = np.mean(self.returns[s])

            iteration += 1

            # if iteration in iterations_to_record:
            #     record(iteration, "On-policy MC with state values, iteration: ", self.V, self.π, self.env.w, self.env.h, self.env.costs)
            if iteration>20000 or np.sum(np.abs(self.V - v)) < 1e-2: break

    def TD0VPrediction(self):
        while True:
            state = self.env.states[np.random.choice(range(len(self.env.states)))]
            # episode:
            while state not in self.env.goals: #not self.env.goalState(state):
                action = random.choice(self.env.actions)# self.policy(state)

                next_state, reward = self.step(state, action)

                G = reward + self.γ*self.apxV.V(next_state)
                self.SGD_update(G, s)
                # self.V[state] += self.α*(reward + self.γ*self.V[next_state] - self.V[state])
                state = next_state
            if iteration>20000:# or np.sum(np.abs(self.V - v)) < 1e-2:
                break

    def LSTD0(self, d):
        Ainv = np.eye(d) / self.ε
        b = np.zeros((d, 1))
        while True: # Looping episodes, loop until convergence
            state = 0# starting state
            x = [0,0]#x(state) #feature representation (column vector)
            while next_state not in self.env.goals:# episode loop
                action = self.sampleΠ(state)
                next_state, reward = self.env.step(state, action)
                next_x = x(next_state)

                v = np.dot( Ainv.T, (x - self.γ * next_x) ).reshape((d, 1))
                Ainv -= np.dot( np.dot( Ainv.T, x ), v.T ) / ( 1 + np.dot( v.T, x ) )
                b += reward * x
                w = np.dot(Ainv, b)# this line would need to update our params
                state = next_state
                x = next_x

    def valueOfPolicy(self):
        total_r = 0
        for s in self.states:
            st = s
            r = -self.costs_average[st]
            while True:
                r += self.immediateReward(st, self.maxAction(st), True)
                if st == self.nextState(st, self.maxAction(st)):
                    break
                st = self.nextState(st, self.maxAction(st))
                if st == goal:
                    r+=100
                    break
            total_r += r
        return total_r

    def explore(self):# returns random action
        return random.choice(self.actions)

    def maxAction(self, s):# returns the action with the highest Q-value
        Qs=[(action, self.Q[(s,action)]) for action in self.actions]
        mx = max(Qs, key=lambda x:x[1])[1]
        mxlt = [a for (a,q) in Qs if q==mx]
        return mxlt[np.random.randint(len(mxlt))]

    def εGreedy(self, state, ε):
        if random.random() < ε:
            return self.explore()
        else:
            return self.maxAction(state)

    def episode(self, τ):
        """ This is center of the algorithm, the agent gets put into a random state,
            then keeps performing actions until it gets to the goal state.

            The Q-table gets updated according to our recusive definition to
            appoximate the true Q-value.
        """
        self.newMask()
        state=(random.randint(0,w-1),random.randint(0,h-1))
        while(True):
            action = self.explore()
#            action = self.εGreedy(state, 0.2)
#            action = self.softmax(state, τ, 1.5)

            r = self.immediateReward(state, action)
            newState = self.nextState(state, action)
            # Update rule:
            # self.Q[(state,action)].q = r + self.γ*self.Q[(newState,self.maxAction(newState))].q
            visits = self.Q[(state,action)].visits + 1
#            prevQ = self.Q[(state,action)].q
            deterministicUpdate = r + self.γ*self.Q[(newState,self.maxAction(newState))].q
            α_n = 1/(1+visits)
            self.Q[(state,action)].q = (1 - α_n)*self.Q[(state,action)].q + α_n*deterministicUpdate
            self.Q[(state,action)].visits = visits
            if newState == self.goal:
                break
            state = newState

    def runNEpisodes(self, n, iterations_to_record):# runs n episodes and records at the list of iterations
        τ = 100

        for iteration in range(1, n+1):
            # print("iteration "+str(iteration))

            print(τ)
            self.episode(τ)
            τ*=0.9995

            # log total current reward, go from each starting point and follow policy
#            print("logging...")
            self.loggedRewards.append( ( iteration, self.valueOfPolicy() ) )#sum( [self.Q[(s, self.maxAction(s))].q for s in self.states] ) ) )
#            print("logging done.")

            if iteration == iterations_to_record[0]:
                _ = iterations_to_record.pop(0)
                X, Y, U, V = makeUVM(self.Q, self.w, self.h)
                fn = self.plotroot+"/Q"+str(iteration)+".png"
                title = "Policy visualisation. Iteration: "+str(iteration)
                # plotPolicy(X, Y, U, V, self.costs_average, self.w, self.h, show=False, filename=fn, title=title, cbarlbl="Costs")
                plotCts(costAll, self.w, self.h, show=False, filename=fn, title=title)
