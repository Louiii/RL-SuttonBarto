from modules import *

class BairdsCounterExample:
    """ Environment, small MPD """
    def __init__(self):
        self.states  = [1,2,3,4,5,6,7]
        self.actions = ["solid", "dashed"]

    def step(self, state, action):
        """ returns next_state, reward
        if action=="solid" then always go to state 7
        else action=="dashed": go to any of 1-6 with equal prob
        """
        if action=="solid": return 7, 0
        return np.random.choice(self.states[:-1]), 0

class ValueFn:
    def __init__(self, α=0.01, n_states=7):
        self.θ = np.ones(8)
        self.θ[6] = 10
        self.α = α#/n_states

    def x(self, state):
        """ get feature vector of state """
        x_ = np.zeros(len(self.θ))
        if state in [1,2,3,4,5,6]:
            x_[7] = 1
            x_[state-1] = 2
        elif state==7:
            x_[7] = 2
            x_[state-1] = 1
        return x_

    def value(self, state):
        """ linear function approximation,
        deriv wrt θ == x (derivative of linear value function wrt θ is the feature vector)
        """
        return np.dot(self.θ, self.x(state))

    def update(self, error, state):
        """ self.x(state) == derivative of linear value function wrt θ is the feature vector
        self.α is the step size
        """
        self.θ += self.α * error * self.x(state)


class Agent:
    def __init__(self, env, α=0.01, β=0.005):
        self.env = env
        self.V = ValueFn(α)
        self.γ = 0.99
        self.β = β
        self.Π = self.compute_projection_matrix()

    def b(self, state):
        if np.random.rand() < 6/7 : return "dashed"
        return "solid"

    def semiGradientOffPolicyTD_iteration(self, state):
        action = self.b(state)
        next_state, reward = self.env.step(state, action)

        ρ = 0 if action=="dashed" else 7

        error = ρ*(reward + self.γ*self.V.value(next_state) - self.V.value(state))

        self.V.update(error, state)
        return next_state, reward

    def semiGradientDP_sweep(self):
        error = 0
        for s in self.env.states:
            # only state 7 relevant
            expected_return = 0 + self.γ * self.V.value(7)
            # error += expected_return - self.V.value(s)
            BE = expected_return - self.V.value(s)
            error += BE * self.V.x(s)
            # self.V.update(error, s)
        self.V.θ += error*self.V.α/len(self.env.states)

    def GTD0(self, state, v_t):
        action = self.b(state)
        next_state, reward = self.env.step(state, action)

        ρ = 0 if action=="dashed" else 7

        δ = reward + self.γ * self.V.value(next_state) - self.V.value(state)
        self.V.θ += self.V.α * ρ * ( δ*self.V.x(state) - self.γ*np.dot(self.V.x(next_state), np.dot(self.V.x(state), v_t)) )
        # # self.V.update(error, state)
        v_t += self.β * ρ * (δ - np.dot(self.V.x(state), v_t)) * self.V.x(state)
        return next_state, reward

    def expected_GTD0(self, v_t):
        for s in self.env.states:
            # only state 7 has non-zero importance ratio
            δ = 0 + self.γ * self.V.value(7) - self.V.value(s)
            ρ = 7
            # Under behavior policy, state distribution is uniform, so the probability for each state is 1.0 / len(STATES)
            expected_update_theta = 1.0 / len(self.env.states) * 1/7 * ρ * (δ * self.V.x(s) - self.γ * self.V.x(7) * np.dot(v_t, self.V.x(s)) )
            self.V.θ += self.V.α * expected_update_theta
            expected_update_v_t1 = 1.0 / len(self.env.states) * 1/7 * ρ * (δ - np.dot(v_t, self.V.x(s))) * self.V.x(s)
            v_t += self.β * expected_update_v_t1

    def expected_emphatic_TD(self, M):
        # M is the emphasis
        I = 1# interest
        # synchronous update (θ, M)
        expected_θ_update = 0
        expected_next_M = 0
        for s in self.env.states:
            ρ = 7 if s == 7 else 0# 1/p else 0
            nextM = self.γ * ρ * M + I
            expected_next_M += nextM
            # only state 7 has non-zero importance ratio
            δ = 0 + self.γ * self.V.value(7) - self.V.value(s)
            expected_θ_update += 1.0 / len(self.env.states) * nextM * δ * self.V.x(s)
        self.V.θ += self.V.α * expected_θ_update
        return expected_next_M / len(self.env.states)

    def compute_RMSVE(self):
        X = np.array([self.V.x(s) for s in self.env.states])
        state_distribution = np.ones(7) / 7
        return np.sqrt(np.dot(np.dot(X, self.V.θ)**2, state_distribution))

    def compute_RMSPBE(self):
        """
        δ_w(s) = E_π[ R + γV_w(s') - V_w(s)]
        BE(w)  = |δ_w|^2
        PBE(w) = |Πδ_w|^2 =
        """
        state_distribution = np.ones(7) / 7
        δ = {s:0 for s in self.env.states}#np.zeros(len(self.env.states))
        for state in self.env.states:
            for next_state in self.env.states:
                if next_state == 7:
                    δ[state] += 0 + self.γ * self.V.value(next_state) - self.V.value(state)
        PBE = np.dot(self.Π, np.array(list(δ.values())))**2
        RMS_PBE = np.sqrt(np.dot(PBE, state_distribution))
        return RMS_PBE

    def compute_projection_matrix(self):
        X = np.diag(np.array([2 for _ in range(len(self.env.states))]))
        X = np.c_[X, np.ones(len(self.env.states))]
        X[6, 6] = 1
        X[6, 7] = 2
        X = np.matrix(X)

        # state distribution for the behavior policy
        state_dist = np.ones(7) / 7
        D = np.matrix(np.diag(state_dist))
        # projection matrix for minimize MSVE:
        return np.asarray(X * np.linalg.pinv( X.T * D * X ) * X.T * D)
