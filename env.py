import gym
import numpy as np
import  matplotlib.pyplot as plt

env = gym.make('Taxi-v3')

state = env.reset()
for t in range(100):
    action = env.action_space.sample() #Biramo nasumicno akciju iz action_space
    plt.axis('off')
    state, reward, done, _ = env.step(action) # izvrsavanje akcije
    env.render() # Stampanje okruzenja
    if done:
        print('Score: ', t+1)
        break
        
env.close()