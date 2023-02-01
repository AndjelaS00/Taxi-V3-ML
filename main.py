from agent import Agent
from monitor import interact
import gym
import numpy as np
import  matplotlib.pyplot as plt

env = gym.make('Taxi-v3') # Ucitavanje okruzenja
agent = Agent()  # Kreiranje instance agent
avg_rewards, best_avg_reward = interact(env, agent) # Treniranje agenta

#Prikazivanje igre 
def render():
    state = env.reset()
    score = 0
    while True:
        env.render()
        action = agent.select_action(state, 0,  env)
        state2,reward,done,info = env.step(action)
        score+=reward
        if done:
            break
        state = state2
    print('The score of the game -- {}'.format(score))

render()