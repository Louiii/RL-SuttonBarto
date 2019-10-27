import numpy as np



class Blackjack:
	def __init__(self):
		self.cards = list(range(1, 11))# A,2,3,..,10
		self.states = [i for i in range(2, 22)]
		self.goals = ['lost', 'won']
		self.actions = ['hit', 'stick']
		self.dealer_π = {s:'stick' if s>=17 else 'hit' for s in self.states}
		self.reset()

	def reset(self):
		self.dealers_hand = None

	def pick_up():
		return min(10, np.random.randint(1, 14))

	def step(self, state, action):
		# if len(state)==2:
		# 	# check for natural
		# 	if 1 in state:

		# 	if sum()

		# return next_state, action