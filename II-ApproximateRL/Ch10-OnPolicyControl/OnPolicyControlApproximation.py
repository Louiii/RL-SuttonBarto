from modules import *

class QTiling:
    def __init__(self, env, α, num_of_tilings, max_size):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings
        self.α = α / num_of_tilings
        self.θ = np.zeros(max_size)# weight for each tile
        self.env = env
        self.hash_table = IHT(max_size)

    def x(self, state, action):
        x_ = np.zeros(len(self.θ))
        features = self.F(state, action)
        for i in features:
            x_[i] = 1
        return x_

    def value(self, state, action):
        # return np.dot(self.x(state, action), self.θ)
        return np.sum(self.θ[self.F(state, action)])


class MountainCarQ(QTiling):
    def __init__(self, env, α, num_of_tilings=8, max_size=2048):
        super().__init__(env, α, num_of_tilings, max_size)
        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (env.pos_bound[1] - env.pos_bound[0])
        self.velocity_scale = self.num_of_tilings / (env.vel_bound[1] - env.vel_bound[0])

    def F(self, state, action):
        position, velocity = state
        return tiles(self.hash_table, self.num_of_tilings, [self.position_scale * position, self.velocity_scale * velocity], [action])

    def value(self, state, action):
        # return np.dot(self.x(state, action), self.θ)
        position, velocity = state
        if position == self.env.pos_bound[1]: return 0.0
        return np.sum(self.θ[self.F(state, action)])

    def cost_to_go(self, state):
        return -np.max([self.value(state, action) for action in self.env.actions])

class ServersExampleQ(QTiling):
    def __init__(self, env, α, num_of_tilings, β=0.01):
        super().__init__(env, α, num_of_tilings, 2048)
        # state features needs scaling to satisfy the tile software
        self.server_scale = self.num_of_tilings / float(env.n_servers)
        self.priority_scale = self.num_of_tilings / float(len(env.priorities) - 1)

        self.average_reward = 0.0
        self.β = β

    def F(self, state, action):
        '''feature encoding'''
        priority, free_servers = state
        active_tiles = tiles(self.hash_table, self.num_of_tilings,
                            [self.server_scale * free_servers, self.priority_scale * priority],
                            [action])
        return active_tiles

    # estimate the value of given state without subtracting average
    def state_value(self, state):
        priority, free_servers = state
        values = [self.value(state, action) for action in self.env.actions]
        # if no free server, can't accept
        if free_servers == 0: return values[0]
        return np.max(values)

class Agent:
    def __init__(self, env, α=0.3, β=0.01, γ=1, ε=0.0, n=1, servers_example=False, QFnArgs=[]):
        self.env  = env
        self.apxQ = MountainCarQ(env, *QFnArgs) if not servers_example else ServersExampleQ(env, *QFnArgs)

        self.α = self.apxQ.α
        self.β = β
        self.γ = γ
        self.ε = ε
        self.n = n
        # Buffer
        self.S = {i: None for i in range(n+1)}
        self.A = {i: None for i in range(n+1)}
        self.R = {i: None for i in range(n+1)}

        self.avr_R = 0

    def maxAction(self, s):# returns the action with the highest Q-value
        Qs=[(action, self.apxQ.value(s,action)) for action in self.env.actions]
        # return max(Qs, key=lambda x:x[1])[0]
        mx = max(Qs, key=lambda x:x[1])[1]
        mxlt = [a for (a,q) in Qs if q==mx]
        return mxlt[np.random.randint(len(mxlt))]

    def εGreedy(self, state):
        if np.random.rand() < self.ε:
            return np.random.choice(self.env.actions)
        return self.maxAction(state)

    def semiGradient_sarsa_episode(self, start):
        state = start
        action = self.εGreedy(state)
        while True:
            next_state, reward = self.env.step(state, action)
            if next_state in self.env.goals:
                δ = self.α * (reward - self.apxQ.value(state, action))
                # grad = self.apxQ.x(state, action)# since linear
                # self.apxQ.θ += δ*grad
                ## the two lines below are faster than the two lines above
                for i in self.apxQ.F(state, action): 
                    self.apxQ.θ[i] += δ
                break
            
            next_action = self.εGreedy(next_state)
            δ = self.α * (reward + self.γ*self.apxQ.value(next_state, next_action) - self.apxQ.value(state, action))
            # grad = self.apxQ.x(state, action)# since linear
            # self.apxQ.θ += δ*grad
            ## the two lines below are faster than the two lines above
            for i in self.apxQ.F(state, action): 
                self.apxQ.θ[i] += δ
            state  = next_state
            action = next_action

    def semiGradient_nStep_sarsa_episode(self, start=(np.random.uniform(-0.6, -0.4), 0)):
        self.S[0] = start
        self.A[0] = self.εGreedy(start)

        t, τ = 0, 0 # track the time (and update time)
        T = np.inf # the length of this episode
        while τ != T - 1:
            if t < T:
                state  = self.S[t%(self.n+1)]
                action = self.A[t%(self.n+1)]
                next_state, reward = self.env.step(state, action)

                # Store in buffer...
                self.S[(t+1)%(self.n+1)] = next_state
                self.R[(t+1)%(self.n+1)] = reward

                if self.env.goal(next_state):#done:#next_state in self.env.goals:
                    T = t+1
                else:
                    self.A[(t+1)%(self.n+1)] = self.εGreedy(next_state)
            τ = t - self.n + 1 # get the time of the state to update
            if τ >= 0:
                # calculate corresponding rewards, G is our returns
                G = sum([self.R[i%(self.n+1)] * self.γ**(i-τ-1) for i in range(τ+1, min(τ+self.n, T)+1)])
                if τ + self.n < T:
                    G += self.apxQ.value(self.S[(τ+self.n)%(self.n+1)], self.A[(τ+self.n)%(self.n+1)]) * self.γ**self.n

                state_to_update, action_to_update = self.S[τ%(self.n+1)], self.A[τ%(self.n+1)]
                # update the value function
                if not self.env.goal(state_to_update):#not state_to_update in self.env.goals:# goals have V = 0
                    δ = self.α * (G - self.apxQ.value(state_to_update, action_to_update))
                    # grad = self.apxQ.x(state_to_update, action_to_update)# since linear
                    # self.apxQ.θ += δ*grad
                    ## the two lines below are faster than the two lines above
                    for i in self.apxQ.F(state_to_update, action_to_update): 
                        self.apxQ.θ[i] += δ
            t += 1 # go to next time step
            state = next_state
        return T

    def differential_SemiGradient_sarsa_control_episode(self, max_steps):
        state = self.env.start
        action = self.εGreedy(state)
        freq = np.zeros(self.env.n_servers+1)
        for i in range(max_steps):
            freq[state[1]] += 1
            next_state, reward = self.env.step(state, action)
            next_action = self.εGreedy(next_state)

            # update
            δ = reward - self.avr_R + self.apxQ.value(next_state, next_action) - self.apxQ.value(state, action)
            self.avr_R += self.β * δ
            # grad = self.apxQ.x(state, state)# since linear
            # self.apxQ.θ += δ*grad*self.α
            ## the two lines below are faster than the two lines above
            for i in self.apxQ.F(state, action): 
                self.apxQ.θ[i] += δ*self.α

            state = next_state
            action = next_action
        print('Frequency of number of free servers:')
        print(freq / max_steps)

    def differential_semiGradient_nStep_sarsa_episode(self, start):
        self.S[0] = start
        self.A[0] = self.εGreedy(start)

        t, τ = 0, 0 # track the time (and update time)
        T = np.inf # the length of this episode
        while τ != T - 1:
            state  = self.S[t%(self.n+1)]
            action = self.A[t%(self.n+1)]
            next_state, reward = self.env.step(state, action)

            # Store in buffer...
            self.S[(t+1)%(self.n+1)] = next_state
            self.R[(t+1)%(self.n+1)] = reward

            self.A[(t+1)%(self.n+1)] = self.εGreedy(next_state)

            τ = t - self.n + 1 # get the time of the state to update
            if τ >= 0:
                # calculate corresponding rewards, G is our returns
                δ = sum([self.R[i%(self.n+1)]  for i in range(τ+1, τ+self.n+1)])
                δ += n * (self.apxQ(self.S[(τ+self.n)%(self.n+1)], self.A[(τ+self.n)%(self.n+1)]) - self.apxQ(self.S[τ%(self.n+1)], self.A[τ%(self.n+1)]) -self.avr_R)
                self.avr_R += self.β * δ
                # grad = self.apxQ.x(self.S[(τ+self.n)%(self.n+1)], self.A[(τ+self.n)%(self.n+1)])# since linear
                # self.apxQ.θ += δ*grad*self.α
                ## the two lines below are faster than the two lines above
                for i in self.apxQ.F(self.S[(τ+self.n)%(self.n+1)], self.A[(τ+self.n)%(self.n+1)]): 
                    self.apxQ.θ[i] += δ*self.α
            t += 1 # go to next time step
            state = next_state
        return T

