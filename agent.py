import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6 , alpha = 0.5 , gamma = 0.6):
        self.nA = nA //akcije 
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state, eps,env):
        if(random.random() > eps):     # Epsilon greedy policy definition
            return (np.argmax(self.Q[state]))
        else:
            return (random.choice(np.arange(env.action_space.n))) 
        
    def update_Q(self , alpha ,gamma ,  Q , state , action , reward , next_state = None):
            current = self.Q[state][action]
            Qsa_next = np.max(Q[next_state,:]) if next_state is not None else 0 # Q-Learning 
            target = reward + (gamma * Qsa_next)
            new_value = current + (alpha * (target - current))
            return new_value

    def step(self, alpha  , gamma , state, action, reward, next_state, done):
        self.Q[state][action]  += self.update_Q(self , alpha , gamma , self.Q  ,state , action , reward , next_state) # Update