import gym
import numpy as np

class QValueFunction:
    def __init__(self, env, α, num_of_tilings=8, max_size=2048):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings
        self.α = α / num_of_tilings# divide step size equally to each tiling

        self.hash_table = IHT(max_size)

        self.θ = np.zeros(max_size)# weight for each tile

        self.env = env

        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (env.pos_bound[1] - env.pos_bound[0])
        self.velocity_scale = self.num_of_tilings / (env.vel_bound[1] - env.vel_bound[0])

    # get indices of active tiles for given state and action
    def get_active_tiles(self, state, action):
        position, velocity = state
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        active_tiles = tiles(self.hash_table, self.num_of_tilings,
                            [self.position_scale * position, self.velocity_scale * velocity],
                            [action])
        return active_tiles

    # estimate the value of given state and action
    def value(self, state, action):
        position, velocity = state
        if position == self.env.pos_bound[1]: return 0.0
        active_tiles = self.get_active_tiles(state, action)
        return np.sum(self.θ[active_tiles])

    def F(self, state, action):
        return self.get_active_tiles(state, action)

    def x(self, state, action):
        x_ = np.zeros(len(self.θ))
        features = self.F(state, action)
        for i in features:
            x_[i] = 1
        return x_

class tilecoder:

    def __init__(self, env, numTilings, tilesPerTiling):
        # self.maxIn = env.observation_space.high
        self.maxIn = [3, 3.5, 0.25, 3.5]
        #self.minIn = env.observation_space.low
        self.minIn = [-3, -3.5, -0.25, -3.5]
        self.numTilings = numTilings
        self.tilesPerTiling = tilesPerTiling
        self.dim = len(self.maxIn)
        self.numTiles = (self.tilesPerTiling**self.dim) * self.numTilings
        self.actions = env.action_space.n
        self.n = self.numTiles * self.actions
        self.tileSize = np.divide(np.subtract(self.maxIn,self.minIn), self.tilesPerTiling-1)
        self.θ = np.random.uniform(-0.001, 0, size=(self.n))

    def getFeatures(self, state):
        ''' Returns indices of the tiles '''
        ### ENSURES LOWEST POSSIBLE INPUT IS ALWAYS 0
        self.state = np.subtract(state, self.minIn)
        tileIndices = np.zeros(self.numTilings)
        matrix = np.zeros([self.numTilings,self.dim])
        for i in range(self.numTilings):
            for i2 in range(self.dim):
                matrix[i,i2] = int(self.state[i2] / self.tileSize[i2] + i / self.numTilings)
        for i in range(1,self.dim):
            matrix[:,i] *= self.tilesPerTiling**i
        for i in range(self.numTilings):
            tileIndices[i] = (i * (self.tilesPerTiling**self.dim) + sum(matrix[i,:]))
        return tileIndices

    def F(self, state, action):
        features = self.getFeatures(state)
        features_action = []
        for i in features:
            features_action.append( int(i + (self.numTiles*action)) )
        return np.array(features_action)

    def oneHotVector(self, features, action):
        oneHot = np.zeros(self.n)
        for i in features:
            index = int(i + (self.numTiles*action))
            oneHot[index] = 1
        return oneHot

    def x(self, state, action):
        F = self.getFeatures(state)
        return self.oneHotVector(F, action)

    def getVal(self, features, action):
        val = 0
        for i in features:
            index = int(i + (self.numTiles*action))
            val += self.θ[index]
        return val

    def value(self, state, action):
        F = self.getFeatures(state)
        Q = self.getQ(F)
        return Q[action]

    def getQ(self, features):
        Q = np.zeros(self.actions)
        for i in range(self.actions):
            Q[i] = self.getVal(features, i)
        return Q

class CartPole:
    def __init__(self, α, λ, γ, ε):
        self.α = α
        self.λ = λ
        self.γ = γ
        self.ε = ε
        self.env = gym.make('CartPole-v0')
        self.Q = tilecoder(self.env, 4,22)
        self.actions = list(range(self.env.action_space.n))

    def episode(self, max_steps):
        state = env.reset()
        for t in range(max_steps):
            # env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done: return t+1

    def render_episode(self, env, max_steps):
        state = env.reset()
        for t in range(max_steps):
            env.render()
            action = self.εGreedy(state, self.ε)
            state, reward, done, info = env.step(action)
            if done: return t+1

    def maxAction(self, s):# returns the action with the highest Q-value
        # Qs=[(action, self.Q.value(s,action)) for action in self.env.actions]
        # mx = max(Qs, key=lambda x:x[1])[1]
        # mxlt = [a for (a,q) in Qs if q==mx]
        # i = np.random.randint(len(mxlt))
        # mxlt[i]
        Qs = { a: self.Q.value(s, a) for a in self.actions }
        return max(Qs, key=Qs.get)

    def εGreedy(self, state, ε):
        if np.random.rand() < ε:
            return np.random.choice(self.actions)#self.env.action_space.sample
        return self.maxAction(state)

    def sarsaλ(self, max_steps, replacing_traces=True):
        state = self.env.reset()
        action = self.εGreedy(state, self.ε)
        z = np.zeros(len(self.Q.θ))
        step = 0
        while step < max_steps:
            step += 1
            next_state, reward, done, info = self.env.step(action)
            # self.env.render()
            δ = reward

            for i in self.Q.F(state, action):
                δ -= self.Q.θ[i]
                if replacing_traces:
                    z[i] = 1
                else:# accumulating trace
                    z[i] += 1
            if done:
                self.Q.θ += self.α * δ * z
                break
            next_action = self.εGreedy(next_state, self.ε)
            for i in self.Q.F(next_state, next_action):
                δ += self.γ * self.Q.θ[i]

            self.Q.θ += self.α * δ * z
            z *= self.γ * self.λ

            state = next_state
            action = next_action
        return step
