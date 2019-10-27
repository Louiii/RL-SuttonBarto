from modules import *

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

    def cost_to_go(self, position, velocity):
        costs = []
        for action in self.env.actions:
            costs.append(self.value((position, velocity), action))
        return -np.max(costs)


class QValueFn:
    def __init__(self, env, α=0.05):
        self.goals = env.goals
        self.n_θ = len(env.states)*len(env.actions)
        self.θ = np.zeros(self.n_θ)
        self.α = α
        n = len(env.states)
        self.X = {(env.states[i], env.actions[j]):[i+j*n] for i in range(len(env.states)) for j in range(len(env.actions))}

    def F(self, state, action):
        return self.X[(state, action)]

    def x(self, state, action):
        x_ = np.zeros(self.n_θ)
        features = self.F(state, action)
        for i in features:
            x_[i] = 1
        return x_

    def value(self, state, action):
        features = self.X[(state, action)]
        return sum([self.θ[f] for f in features])# return self.θ[self.X[(state, action)]#np.dot(self.x(state, action), self.θ)

class PWQ:
    def __init__(self, env, α):
        num_of_tilings = 8
        self.ns, self.na = len(env.states), len(env.actions)
        print(self.ns*self.na)
        self.θ = np.zeros(self.ns*self.na)
        self.α = α / num_of_tilings

    def F(self, state, action):
        r, c = state
        return [int(self.na * (r + c * self.ns**0.5) + action)]

    def value(self, state, action):
        return self.θ[self.F(state, action)]

class SAR:
    def __init__(self, S=None, A=None, R=None):
        self.S = S
        self.A = A
        self.R = R
max_steps = 3000

class Agent:
    def __init__(self, env, γ, λ, QFnArgs, pw=False):
        self.env = env
        self.Q = QValueFunction(env, *QFnArgs) if not pw else PWQ(env, *QFnArgs)
        # self.Q = QValueFn(env, α)
        self.γ = γ# discount factor
        self.λ = λ# trace decay rate

    def maxAction(self, s):
        Qs = { a: self.Q.value(s, a) for a in self.env.actions }
        return max(Qs, key=Qs.get)

    def εGreedy(self, state, ε):
        if np.random.rand() < ε:
            return np.random.choice(self.env.actions)
        return self.maxAction(state)

    def sarsaλ(self, replacing_traces=True, clearing=False, start=(np.random.uniform(-0.6, -0.4), 0)):
        ε = 0.0
        state, action = start, self.εGreedy(start, ε)
        z = np.zeros(len(self.Q.θ))
        step, total_reward = 0, 0
        while step < max_steps:
            step += 1
            next_state, reward = self.env.step(state, action)
            δ = reward

            if clearing:
                for a in self.env.actions:
                    z[self.Q.F(state, a)] = 0
            for i in self.Q.F(state, action):
                δ -= self.Q.θ[i]
                if replacing_traces:
                    # if clearing: # clear trace of other actions!
                    #     features_to_clear =
                    #     z = np.zeros(len(self.Q.θ))
                    z[i] = 1
                else:# accumulating trace
                    z[i] += 1
            if self.env.goal(next_state):#next_state in self.env.goals:
                self.Q.θ += self.Q.α * δ * z
                break
            next_action = self.εGreedy(next_state, ε)
            for i in self.Q.F(next_state, next_action):
                δ += self.γ * self.Q.θ[i]

            self.Q.θ += self.Q.α * δ * z
            z *= self.γ * self.λ

            state = next_state
            action = next_action
            # print(step)
            total_reward += reward
        return step, total_reward

    def trueOnline_sarsaλ(self, start=(np.random.uniform(-0.6, -0.4), 0)):
        ε = 0.0
        state, action = start, self.εGreedy(start, ε)
        x = self.Q.x(state, action)
        z = np.zeros(len(self.Q.θ))
        Q_old = 0
        step, total_reward = 0, 0
        while step < max_steps:
            step += 1
            next_state, reward = self.env.step(state, action)
            next_action = self.εGreedy(next_state, ε)
            next_x = self.Q.x(next_state, next_action)

            Q = np.dot(x, self.Q.θ)
            next_Q = np.dot(next_x, self.Q.θ)

            δ = reward + self.γ * next_Q - Q
            z = self.γ * self.λ * z + (1 - self.Q.α * self.γ * self.λ * np.dot(z, x)) * x
            self.Q.θ += self.Q.α * (δ + Q - Q_old) * z - self.Q.α * (Q - Q_old) * x

            Q_old = next_Q
            x = next_x
            state = next_state
            action = next_action
            # print(step)
            total_reward += reward
            if self.env.goal(next_state): break
        return step, total_reward






#
