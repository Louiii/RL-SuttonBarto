import numpy as np

class MDP_Norm:
	def __init__(self):
		self.states  = ['1', 'A', 'B', '2']
		self.actions = {'A':['left', 'right'], 'B':['a','b','c','d','e','f','g'], '1':[''], '2':['']}
		self.goals   = ['1', '2']
		self.start   = 'A'

	def step(self, state, action):
		if state=='A':
			if action=='right': 
				return '2', 0
			return 'B', 0
		if state=='B': return '1', np.random.normal(-0.1)
		return state, 0