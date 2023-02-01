from collections import deque
import sys
import math
import numpy as np
import agent as a


def interact(env, agent, num_episodes=30000, window=100,alpha = 0.6):
    # inicijalizovati prosecnu nagradu
    avg_rewards = deque(maxlen=num_episodes)
    # inicijalizovati najbolju prosecnu nagradu
    best_avg_reward = -math.inf
    # inicijalizovati monitor za najnovije nagrade
    samp_rewards = deque(maxlen=window)
    # za svaku apizodu
    for i_episode in range(1, num_episodes+1):
        score = 0
        state = env.reset()
        
        eps = 0.6/ i_episode
        
        while True:
            # agent bira akciju
            action = agent.select_action(state,eps,env)
            next_state , reward , done , info = env.step(action)
            score += reward
            agent.Q[state][action] = agent.update_Q(0.5 ,0.6 , agent.Q , state , action , reward , next_state)
            state = next_state
            if done:
                samp_rewards.append(score)
                break
        if (i_episode >= 100):
            # dobijamo prosecnu nagradu za poslednjih 100 epizoda
            avg_reward = np.mean(samp_rewards)
            # dodajemo na deque
            avg_rewards.append(avg_reward)
            # azuriramo najbolju prosecnu nagradu
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # pratiti napredak
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        # proveriti da li je zadatak resem (prema OpenAI Gym)
        if best_avg_reward >= 9.4:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')
    return avg_rewards, best_avg_reward